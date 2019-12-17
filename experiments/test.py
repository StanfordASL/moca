import argparse
import numpy as np
from metacpd.main.dataset import RainbowMNISTDataset, SwitchingSinusoidDataset, MiniImageNet
from metacpd.main.utils import get_prgx, mask_nlls, compute_acc
from metacpd.main import MOCA, ALPaCA, PCOC, ConvNet
import pickle
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


parser = argparse.ArgumentParser(description='Test model')

default_model_path = 'best'
parser.add_argument('--model.model_name', type=str, default=None, metavar='MODELPATH',
                    help="file of pretrained model to evaluate (default: Error)")
parser.add_argument('--data.batch_size', type=int, default=1, metavar='BATCHSIZE',
                    help="meta batch size (default: 1)")
parser.add_argument('--data.horizon', type=int, default=400, metavar='HORIZON',
                    help="test horizon (default: 400)")
parser.add_argument('--data.train_horizon', type=int, default=100, metavar='TRAINHORIZON',
                    help="train horizon (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='TESTEP',
                    help="number of episode to test on (default: 100)")
parser.add_argument('--data.cuda', type=int, default=-1, metavar='CUDA_DEVICE',
                    help='which cuda device to use. if -1, uses cpu')
parser.add_argument('--data.hazard', type=float, default=0.01, metavar='HAZARD',
                    help='hazard (default: 0.01)')
parser.add_argument('--data.train_hazard', type=float, default=0.1, metavar='HAZARD',
                    help='hazard (default: 0.1)')
parser.add_argument('--model.model', type=str, default=None, metavar='model',
                    help="which ablation to use (default: reuse training model)")
parser.add_argument('--model.train_model', type=str, default='main', metavar='trmodel',
                    help="training model to test (default: reuse training model)")
parser.add_argument('--train.task_supervision', type=float, default=None, metavar='LRDECAY',
                    help='Percentage of task switches labeled')
parser.add_argument('--train.train_task_supervision', type=float, default=None, metavar='LRDECAY',
                    help='Percentage of task switches labeled')
parser.add_argument('--train.seed', type=int, default=1000, metavar='SEED',
                    help='numpy seed')
parser.add_argument('--train.experiment_id', type=int, default=0, metavar='SEED',
                    help='unique experiment identifier seed')
parser.add_argument('--data.dataset', type=str, default='Sinusoid', metavar='DS',
                    help="data set name (default: Error)")
parser.add_argument('--data.window_length', type=int, default=20, metavar='WINDOW',
                    help="sliding window length for train model (default: 20)")
parser.add_argument('--data.test_window_length', type=int, default=20, metavar='WINDOW',
                    help="sliding window length for test model (default: 20)")
parser.add_argument('--train.train_experiment_name', type=str, default=None, metavar='SEED',
                    help='name of experiment')
parser.add_argument('--train.experiment_name', type=str, default=None, metavar='SEED',
                    help='name of experiment')
parser.add_argument('--train.oracle_hazard', type=float, default=None, metavar='LR',
                    help='Hazard rate for oracle (curriculum) (default: None)')

def main(updated_config):
    
    load_path = str(updated_config['train.experiment_id']) + '/' + updated_config['train.train_experiment_name'] + '/' + updated_config['data.dataset'] + '/'
    save_path = str(updated_config['train.experiment_id']) + '/' + updated_config['train.experiment_name'] + '/' + updated_config['data.dataset'] + '/'
    if updated_config['model.train_model'] == 'sliding_window':
        if updated_config['data.window_length'] == 0:
            load_path += 'toe/'
        else:
            load_path += updated_config['model.train_model'] + str(updated_config['data.window_length']) + '/'

    else:
        load_path += updated_config['model.train_model'] + '/'
    
    if updated_config['model.model'] is None:
        updated_config['model.model'] = updated_config['model.train_model']
    
    if updated_config['model.model'] == 'sliding_window':
        if updated_config['data.test_window_length'] == 0:
            save_path += 'toe/'
        else:
            save_path += updated_config['model.model'] + str(updated_config['data.test_window_length']) + '/'
    elif updated_config['model.train_model'] == 'conv_net':
        save_path += 'toe/'
    else:
        save_path += updated_config['model.model'] + '/'
    
    if updated_config['train.oracle_hazard'] is None:
        path = 'h' + str(updated_config['data.train_hazard']) + '/'
    else:
        path = 'h' + str(updated_config['train.oracle_hazard']) + '/'
            
    def to_cuda(x):
        if config['data.cuda'] >= 0:
            return x.float().cuda(config['data.cuda'])
        else:
            return x.float()
        
    seed = updated_config['train.seed']

    print('saved_models/' + load_path + path)
    dir_list = next(os.walk('saved_models/' + load_path + path))[1]
    
    for direc in dir_list:
        with torch.no_grad():
            if direc not in ['10']:
                np.random.seed(seed)
                torch.manual_seed(seed)

                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                # join the experiment seed with the path
                dir_path = path + direc
                print('Test model path: ', save_path + dir_path)

                if updated_config['model.model_name'] is None:
                    raise ValueError('must specify model name')

                checkpoint = torch.load('saved_models/' + load_path + dir_path  + '/' + updated_config['model.model_name'])
                config = checkpoint['config']

                for k,v in updated_config.items():
                    if v is not None:
                        config[k] = v

                config['train.oracle_hazard'] = None #should take this out

                batch_size = config['data.batch_size']
                horizon = config['data.horizon']

                if config['data.dataset'] == 'MiniImageNet':
                    dataset = MiniImageNet(config,'test')
                    problem_setting = 'class'
                    if config['model.model'] == 'conv_net':
                        model = ConvNet(config)
                        model.load_state_dict(checkpoint['conv_net'])
                    else:
                        meta_learning_model = PCOC(config)
                        meta_learning_model.load_state_dict(checkpoint['meta_learning_model'])
                        model = MOCA(meta_learning_model, config)
                        model.load_state_dict(checkpoint['moca'])

                elif config['data.dataset'] == 'RainbowMNIST':
                    dataset = RainbowMNISTDataset(config,train='test')
                    problem_setting = 'class'
                    config['model.classification'] = True
                    if config['model.model'] == 'conv_net':
                        model = ConvNet(config)
                        model.load_state_dict(checkpoint['conv_net'])
                    else:
                        meta_learning_model = PCOC(config)
                        meta_learning_model.load_state_dict(checkpoint['meta_learning_model'])
                        model = MOCA(meta_learning_model, config)
                        model.load_state_dict(checkpoint['moca'])

                elif config['data.dataset'] == 'Sinusoid':
                    config['model.sigma_eps'] = '[.05]' # todo move to config
                    dataset = SwitchingSinusoidDataset(config)
                    problem_setting = 'reg'

                    meta_learning_model = ALPaCA(config)
                    meta_learning_model.load_state_dict(checkpoint['meta_learning_model'])
                    model = MOCA(meta_learning_model, config)
                    model.load_state_dict(checkpoint['moca'])

                elif config['data.dataset'] == 'NoiseSinusoid':
                    dataset = SwitchingNoiseSinusoidDataset(config)
                    problem_setting = 'reg'

                    meta_learning_model = t_ALPaCA(config)
                    meta_learning_model.load_state_dict(checkpoint['meta_learning_model'])
                    model = MOCA(meta_learning_model, config)
                    model.load_state_dict(checkpoint['moca'])

                else:
                    raise NotImplementedError    

                model = to_cuda(model)
                model = model.eval()

                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

                running_nll = []
                running_acc = []
                timings_mode = False
                running_times = np.zeros((config['data.horizon'], config['data.test_episodes']))

                for i, data in enumerate(dataloader):
                    if data['x'].shape[0] == config['data.batch_size']:
                        if i >= config['data.test_episodes']:
                            break
                        
                        inputs = to_cuda(data['x'])
                        labels = to_cuda(data['y'])
                        switch_times = to_cuda(data['switch_times'])

                        prgx, task_sup = get_prgx(config,horizon,batch_size,switch_times=switch_times)

                        # moca
                        if timings_mode:
                            _,_,nlls, timing_array = model(inputs,labels,prgx=prgx, task_supervision=task_sup, return_timing=timings_mode)
                        else:
                            _,_,nlls = model(inputs,labels,prgx=prgx, task_supervision=task_sup, return_timing=timings_mode)
                            
                        if not timings_mode:
                            if problem_setting == 'class':
                                # nlls are shape batchsize x t x n_classes, so we need to mask
                                # for loss and also eval accuracy
                                accs = compute_acc(labels, nlls) # batchsize x t
                                nlls = mask_nlls(labels, nlls) # now batchsize x t

                            mean_nll = torch.mean(nlls)

                            running_nll.append(mean_nll.item())
                            if problem_setting == 'class':
                                mean_acc = torch.mean(accs)
                                running_acc.append(mean_acc.item())

                        else:
                            running_times[:,i] = timing_array

                # compute stats on NLL and accuracy

                if not timings_mode:
                    mean_nll = np.mean(running_nll)
                    nll_stderr = np.std(running_nll)/np.sqrt(len(running_nll))

                    print('Mean NLL: ', mean_nll)
                    print('NLL 95% confidence: +/-', nll_stderr*1.96)

                    results = {
                        'running_nll': running_nll,
                        'mean_nll': mean_nll,
                        'nll_conf': nll_stderr*1.96
                    }
                else:
                    results = {
                        'timings': running_times
                    }

                if problem_setting == 'class' and not timings_mode:
                    mean_acc = np.mean(running_acc)
                    acc_stderr = np.std(running_acc)/np.sqrt(len(running_acc))

                    print('Mean accuracy: ', 100.*mean_acc)
                    print('Accuracy 95% confidence: +/-', acc_stderr*196.)

                    results['running_acc'] = running_acc
                    results['mean_acc'] = 100.*mean_acc
                    results['acc_conf'] = 196.*acc_stderr

                print('---------------\n')

                # saving results
                filename = 'results/' + save_path + dir_path + '/' + updated_config['model.model_name'][:-3] + '.pickle'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'wb') as f:
                    pickle.dump(results, f)

args = vars(parser.parse_args())
main(args)
