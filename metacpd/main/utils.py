import numpy as np
import torch
import os
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
def conv_block(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,3,padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )

def final_conv_block(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,3,padding=1),
#                         nn.BatchNorm2d(out_channels),
                        nn.MaxPool2d(2)
                        )

def listdir_nohidden(path):
    dir_list = []
    for f in os.listdir(path):
        if not f.startswith('.'):
             dir_list.append(f)
    return dir_list


def mask_nlls(y,likelihoods):
    """
    y: onehot labels: shape (..., n_classes)
    likelihood: per class: shape (..., n_classes)
    """
    # mask with y
    return torch.sum(y * likelihoods,dim=-1)

def compute_acc(y,nlls):
    # compute accuracy using nlls
    pred_class = torch.argmin(nlls,-1,keepdim=True)
    
    acc = y.gather(-1, pred_class).squeeze(-1)
    return acc


def get_prgx(config,horizon,batch_size,switch_times=None):

    model = config['model.model']
    sliding_window_len = config['data.window_length']

    if model == 'main' or model == 'conv_net':
        return None, None

    prgx = []
    task_sup = []
    last_switch = np.zeros(batch_size,dtype=int)

    for t in range(horizon):
        prgx_t = np.zeros((batch_size,t+1))
        task_supervision = np.zeros(batch_size)
        for i in range(batch_size):
            if model == 'sliding_window':
                prgx_t[i,max(t-sliding_window_len,0)] = 1
            elif model == 'no_task_change':
                prgx_t[i,t] = 1
            elif model == 'oracle':
                if switch_times[i,t] > 0.5:
                    last_switch[i] = t
                    if config['train.task_supervision'] is not None:
                        if np.random.rand() < config['train.task_supervision']:
                            task_supervision[i] = 1.
                            epsilon = 1e-5
                            prgx_t[i,:] = np.ones(t+1)*epsilon
                            prgx_t[i,last_switch[i]] = 1. - epsilon*t
                            
                            if config['train.oracle_hazard'] is not None:
                                raise NotImplementedError
                                
                if config['train.task_supervision'] is None:
                    if config['train.oracle_hazard'] is not None:
                        if last_switch[i] != 0:
                            prgx_t[i,0] = config['train.oracle_hazard']
                            prgx_t[i,last_switch[i]] = 1. - config['train.oracle_hazard']
                        else:
                            prgx_t[i,last_switch[i]] = 1.
                    else:
                        prgx_t[i,last_switch[i]] = 1.
                
            else:
                raise ValueError('make sure specified model is implemented')


        prgx_t = torch.tensor(prgx_t).float()
        task_supervision = torch.tensor(task_supervision).float()


        if config['data.cuda'] >= 0:
            prgx_t = prgx_t.cuda(config['data.cuda'])
            task_supervision = task_supervision.cuda(config['data.cuda'])

        
        prgx.append(prgx_t)
        task_sup.append(task_supervision)

    if config['train.task_supervision'] is None:
        return prgx, None
    else:
        return prgx, task_sup
    