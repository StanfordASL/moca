import argparse
import numpy as np

from metacpd.main import MOCA, ALPaCA, PCOC, ConvNet
from metacpd.main.encoders import get_encoder

from metacpd.main.dataset import RainbowMNISTDataset, SwitchingSinusoidDataset, MiniImageNet
from metacpd.main.utils import get_prgx, mask_nlls, compute_acc

import torch
from torch import autograd
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.optim as optim
import os


parser = argparse.ArgumentParser(description='Train model')

# ---------- data args
default_dataset = 'Sinusoid'
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
parser.add_argument('--data.batch_size', type=int, default=50, metavar='BATCHSIZE',
                    help="meta batch size (default: 50)")
parser.add_argument('--data.horizon', type=int, default=100, metavar='HORIZON',
                    help="train horizon (default: 100)")
parser.add_argument('--data.hazard', type=float, default=0.1, metavar='HAZARD',
                    help='hazard (default: 0.1)')
parser.add_argument('--data.window_length', type=int, default=10, metavar='WINDOW',
                    help="sliding window length for ablation model (default: 10)")
parser.add_argument('--data.cuda', type=int, default=-1, metavar='CUDA_DEVICE',
                    help='which cuda device to use. if -1, uses cpu')

# ---------- model args
parser.add_argument('--model.model', type=str, default='main', metavar='model',
                    help="which ablation to use (default: main moca model)")
parser.add_argument('--model.x_dim', type=int, default=1, metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=128, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--model.y_dim', type=int, default=1, metavar='YDIM',
                    help="number of classes/dimension of regression label")
parser.add_argument('--model.phi_dim', type=int, default=32, metavar='PDIM',
                    help="dimensionality of embedding space (default: 64)")
parser.add_argument('--model.sigma_eps', type=str, default='[0.05]', metavar='SigEps',
                    help="noise covariance (regression models; Default: 0.05)")
parser.add_argument('--model.Linv_init', type=float, default=0., metavar='Linv',
                    help="initialization of logLinv in PCOC (Default: 0.0)")
parser.add_argument('--model.dirichlet_scale', type=float, default=10., metavar='Linv',
                    help="value of log Dirichlet concentration params (init if learnable; Default: 0.0)")

# ---------- train args
parser.add_argument('--train.train_iterations', type=int, default=7500, metavar='NEPOCHS',
                    help='number of episodes to train (default: 7500)')
parser.add_argument('--train.val_iterations', type=int, default=5, metavar='NEPOCHS',
                    help='number of episodes to validate on (default: 10)')
parser.add_argument('--train.learning_rate', type=float, default=0.02, metavar='LR',
                    help='learning rate (default: 0.02)')
parser.add_argument('--train.decay_every', type=int, default=1500, metavar='LRDECAY',
                    help='number of iterations after which to decay the learning rate')
parser.add_argument('--train.learnable_hazard', type=int, default=0, metavar='learn_hazard',
                    help='enable hazard being learnable')
parser.add_argument('--train.learnable_noise', type=int, default=0, metavar='learn_noise',
                    help='enable noise being learnable (default: false/0)')
parser.add_argument('--train.learnable_dirichlet', type=int, default=0, metavar='learn_dir',
                    help='enable dirichlet concentration being learnable')
parser.add_argument('--train.verbose', type=bool, default=True, metavar='verbose',
                    help='print during training (default: True)')
# parser.add_argument('--train.save_directory', type=str, default='saved_models/most_recent', metavar='save_directory',
#                     help='where model is saved after training (default: False)')
parser.add_argument('--train.grad_accumulation_steps', type=int, default=1, metavar='grad_acc',
                    help='Number of gradient accumulation steps (default: 1)')
parser.add_argument('--train.task_supervision', type=float, default=None, metavar='TASK_SUP',
                    help='Percentage of task switches labeled')
parser.add_argument('--train.seed', type=int, default=1, metavar='SEED',
                    help='numpy seed')
parser.add_argument('--train.experiment_id', type=int, default=0, metavar='SEED',
                    help='unique experiment identifier seed')
parser.add_argument('--train.experiment_name', type=str, default=0, metavar='SEED',
                    help='name of experiment')
parser.add_argument('--train.oracle_hazard', type=float, default=None, metavar='LR',
                    help='Hazard rate for oracle (curriculum) (default: None)')

def main(config):

    path = str(config['train.experiment_id']) + '/' + config['train.experiment_name'] +'/'
    path += config['data.dataset'] + '/'
    if config['model.model'] == 'sliding_window':
        if config['data.window_length'] == 0:
            path += 'toe/'
        else:
            path += config['model.model'] + str(config['data.window_length']) + '/'
    else:
        path += config['model.model'] + '/'
        
    if config['train.oracle_hazard'] is None:
        path += 'h' + str(config['data.hazard']) + '/'
    else:
        path += 'h' + str(config['train.oracle_hazard']) + '/'

    path += str(config['train.seed']) + '/'
    
    def to_cuda(x):
        if config['data.cuda'] >= 0:
            return x.float().cuda(config['data.cuda'])
        else:
            return x.float()

    def save_model(save_name):
        save_path = 'saved_models/' + path + save_name + '.pt' 

        if config['model.model'] == 'conv_net':
            torch.save({'conv_net': model.state_dict(),
                        'config': config
                       }, save_path)
        else:
            torch.save({'meta_learning_model': meta_learning_model.state_dict(),
                        'moca': model.state_dict(),
                        'config': config
                        }, save_path)

    for bool_arg in ['train.learnable_hazard', 'train.learnable_noise', 'train.learnable_dirichlet']:
        config[bool_arg] = bool(config[bool_arg])

    seed = config['train.seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = config['data.batch_size']
    horizon = config['data.horizon']
    
    
    if not os.path.exists('saved_models/' + path[:-1]):
        os.makedirs('saved_models/' + path[:-1] , exist_ok=True)
    
    writer = SummaryWriter('./runs/' + path + datetime.datetime.now().strftime('y%y_m%m_d%d_s%s'))


    if config['data.dataset'] == 'MiniImageNet':
        dataset = MiniImageNet(config,'train')
        validate_dataset = MiniImageNet(config,'val')
        config['model.classification'] = True
        problem_setting = 'class'
        
        if config['model.model'] == 'conv_net':
            model = ConvNet(config)
        else:
            meta_learning_model = PCOC(config)
            model = MOCA(meta_learning_model, config)

    elif config['data.dataset'] == 'RainbowMNIST':
        dataset = RainbowMNISTDataset(config,train='train')
        validate_dataset = RainbowMNISTDataset(config,train='validate')
        config['model.classification'] = True
        problem_setting = 'class'
        
        if config['model.model'] == 'conv_net':
            model = ConvNet(config)
        else:
            meta_learning_model = PCOC(config)
            model = MOCA(meta_learning_model, config)

    elif config['data.dataset'] == 'Sinusoid':
        dataset = SwitchingSinusoidDataset(config)
        validate_dataset = SwitchingSinusoidDataset(config)
        config['model.classification'] = False
        problem_setting = 'reg'
        
        meta_learning_model = ALPaCA(config)
        model = MOCA(meta_learning_model, config)
        
    elif config['data.dataset'] == 'NoiseSinusoid':
        dataset = SwitchingNoiseSinusoidDataset(config)
        validate_dataset = SwitchingNoiseSinusoidDataset(config)
        config['model.classification'] = False
        problem_setting = 'reg'
        
        meta_learning_model = t_ALPaCA(config)
        model = MOCA(meta_learning_model, config)
        
    else:
        raise notImplementedError

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = to_cuda(model)

    optimizer = optim.Adam(model.parameters(), lr=config['train.learning_rate'])
    optimizer.zero_grad()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.5)

    running_nll = []
    last_mean_val_nll = 10000
    running_acc = []

    if problem_setting == 'reg':
        best_val = 10000.
    else:
        best_val = 0.

    for i, data in enumerate(dataloader):
        if data['x'].shape[0] == config['data.batch_size']:

            if (i+1) % config['train.decay_every'] == 0:
                scheduler.step()
                if config['train.verbose']:
                    print('decreasing learning rate\n')
                                
            if i > config['train.train_iterations']:
                break

            inputs = to_cuda(data['x'])
            labels = to_cuda(data['y'])
            switch_times = to_cuda(data['switch_times'])


            prgx, task_supervision = get_prgx(config,horizon,batch_size,switch_times=switch_times)

            _,_,nlls= model(inputs,labels,prgx=prgx, task_supervision=task_supervision)
            if problem_setting == 'class':
                # nlls are shape batchsize x t x n_classes, so we need to mask
                # for loss and also eval accuracy
                accs = compute_acc(labels, nlls) # batchsize x t
                nlls = mask_nlls(labels, nlls) # now batchsize x t

            mean_nll = torch.mean(nlls)
            mean_nll.backward()




            if i % config['train.grad_accumulation_steps'] == 0:
                optimizer.step()

                running_nll.append(mean_nll.item())
                writer.add_scalar('NLL/Train', mean_nll.item(), i)

                if problem_setting == 'class':
                    ep_acc = torch.mean(accs).item()
                    writer.add_scalar('Accuracy/Train', ep_acc, i)
                    running_acc.append(ep_acc)
                    if config['model.model'] != 'conv_net':
                        writer.add_scalar('max_Linv_eigenval', np.mean(np.exp(model.meta_learning_alg.logLinv.data.cpu().numpy()).max(0)),i)
                        writer.add_scalar('min_Linv_eigenval', np.mean(np.exp(model.meta_learning_alg.logLinv.data.cpu().numpy()).min(0)),i)

                        writer.add_scalar('max_q_val', np.mean(model.meta_learning_alg.Q.data.cpu().numpy().max(0)),i)
                        writer.add_scalar('min_q_val', np.mean(model.meta_learning_alg.Q.data.cpu().numpy().min(0)),i)

                        if config['train.learnable_noise']:
                            writer.add_scalar('max_sigeps_eigenval', np.mean(np.exp(model.meta_learning_alg.logSigEps.data.cpu().numpy()).max(0)),i)
                            writer.add_scalar('min_sigeps_eigenval', np.mean(np.exp(model.meta_learning_alg.logSigEps.data.cpu().numpy()).min(0)),i)

                if config['train.learnable_hazard']:
                    writer.add_scalar('hazard: ', model.hazard.item(), i)


            if i % 100 == 0:

                print('Iteration ' + str(i) + '/' + str(config['train.train_iterations']))
                print('Train NLL: ', np.mean(running_nll))
                running_nll = []


                if problem_setting == 'class':
                    train_acc = np.mean(running_acc)
                    print('Train Accuracy: ', train_acc)
                    running_acc = []

                if config['train.val_iterations'] != 0:
                    with torch.no_grad():
                        running_val_nll = []
                        running_val_acc = []
                        for j, val_data in enumerate(val_dataloader):


                            val_inputs = to_cuda(val_data['x'])
                            val_labels = to_cuda(val_data['y'])
                            val_switch_times = to_cuda(val_data['switch_times'])

                            prgx, task_supervision = get_prgx(config,horizon,batch_size,switch_times=val_switch_times)

                            _,_,val_nlls = model(val_inputs, val_labels, prgx=prgx, task_supervision=task_supervision)
                            if problem_setting == 'class':
                                # nlls are shape batchsize x t x n_classes, so we need to mask
                                # for loss and also eval accuracy
                                val_accs = compute_acc(val_labels, val_nlls) # batchsize x t
                                val_nlls = mask_nlls(val_labels, val_nlls) # now batchsize x t

                            val_mean_nll = torch.mean(val_nlls)
                            running_val_nll.append(val_mean_nll.item())

                            if problem_setting == 'class':
                                running_val_acc.append(torch.mean(val_accs).item())

                            if j == config['train.val_iterations']:
                                val_nll = np.mean(running_val_nll)

                                writer.add_scalar('NLL/Validation', val_nll, i)
                                if problem_setting == 'class':
                                    val_acc = torch.mean(val_accs)
                                    writer.add_scalar('Accuracy/Validation', val_acc.item(), i)


                                break

                        if config['train.verbose']:
                            print('Validation NLL: ', np.mean(running_val_nll))

                            # only print for classification
                            if problem_setting == 'class':
                                print('Validation Accuracy: ', val_acc.item())

                        if problem_setting == 'class':
                            # save best model by val accuracy
                            if best_val < val_acc:
                                best_val = val_acc
                                save_model('best')

                        elif problem_setting == 'reg':
                            # save best model by val NLL
                            if best_val < val_nll:
                                best_val = val_nll
                                save_model('best')

                save_model(str(i))

                print('--------------------')

            optimizer.zero_grad()

args = vars(parser.parse_args())
main(args)
