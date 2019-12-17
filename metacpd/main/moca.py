import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import math
from copy import deepcopy

from metacpd.main.utils import mask_nlls


class MOCA(nn.Module):
    """
    Wraps an underlying MetaLearning algorithm to allow training on timeseries of
    sequential examples with discrete task switches that are unlabeled.
    """

    def __init__(self, meta_learning_alg, config):
        super().__init__()

        self.config = deepcopy(config)
        self.x_dim = config['model.x_dim']
        self.y_dim = config['model.y_dim']

        self.classification = config['model.classification']

        self.meta_learning_alg = meta_learning_alg

        # hazard rate:
        hazard_logit = np.log( config['data.hazard'] / (1 - config['data.hazard'] ) )
        self.hazard_logit = nn.Parameter(torch.from_numpy(np.array([hazard_logit])), requires_grad=config['train.learnable_hazard'])

        # initial log_prx:
        self.init_log_prgx = nn.Parameter(torch.zeros([1,1]), requires_grad=False)


    def nll(self, log_pi, log_prgx):
        """
            log_pi: shape(batch_size x t x ...)       log p(new data | x, r=i, data so far) for all i = 0, ..., t
            log_prgx: shape (batch_size x t x ...)    log p(r=i | data so far) for all i = 0, ..., t
        """

        if len(log_pi.shape) == 3:
            return -torch.logsumexp(log_pi + log_prgx.unsqueeze(-1), dim=1)

        return -torch.logsumexp(log_pi + log_prgx, dim=1)

    def log_p_r_given_x(self,log_prx):
        """
            computes log p(r|x)

            inputs: log_prx: (batch_size, t+1), log p(r, x) for each r in 0, ..., t
                    log_prx: (batch_size, t+1), log p(r | x) for each r in 0, ..., t
        """
        return nn.functional.log_softmax(log_prx,dim=1)

    @property
    def log_hazard(self):
        """
        log p( task_switch )
        """
        return torch.log(torch.sigmoid(self.hazard_logit))

    @property
    def log_1m_hazard(self):
        """
        log (1 - p(task_switch))
        """
        return torch.log(1-torch.sigmoid(self.hazard_logit))

    @property
    def hazard(self):
        return torch.sigmoid(self.hazard_logit)

    def forward(self,x_mat,y_mat,prgx=None, task_supervision=None, return_timing=False):
        """
        Takes in x,y batches; loops over horizon to recursively compute posteriors
        Inputs:
        - x_mat; shape = batch size x horizon x x_dim
        - y_mat; shape = batch size x horizon x y_dim
        """
        batch_size = x_mat.shape[0]
        test_horizon = x_mat.shape[1]


        posterior_params_list = []
        log_prgx_list = []
        nll_list = []

        # define initial params and append to list
        # we add a batch dimension and a time dimension
        prior_params = tuple( p[None,None,...] for p in self.meta_learning_alg.prior_params() )


        posterior_params = prior_params
        log_prgx = self.init_log_prgx # p(r, all data so far)

        posterior_params_list.append(posterior_params)


        # start at time t+1
        
        time_array = []
        
        for i in range(test_horizon):
            # grab y, phi:
            
            if return_timing:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
            
            x = x_mat[:,i,:]
            y = y_mat[:,i,:]


            # compute log p(y|x,hist) for all possible run lengths (shape: (batch_size, t+1))
            # and return updated params incorporating new point
            # log_pi_t = log p(y|x,r,hist) for all r = [0,...,i]

            # if classification, log_pi_t == p(y,x|eta) for all y (batchsize, i+1, y_dim)
            # if regression, log_pi_t == p(y|x,eta)
            log_pi_t, updated_posterior_params = self.meta_learning_alg.log_predictive_prob(x[:,None,:], y[:,None,:], posterior_params, update_params=True)
            if self.classification:
                log_pygx =  nn.functional.log_softmax(log_pi_t, dim=-1) # normalize to get p(y | x) # (batchsize, i+1, y_dim)

                # update p(r_t) given just the x value
                log_p_newx_gr = torch.logsumexp(log_pi_t, dim=-1) # sum over all y values, shape (batch_size, i+1)
                log_prgx = log_prgx + log_p_newx_gr # (batch_size, i+1) # log p ( r_{i} \mid x_{0,i}, y_{0,i-1} )
                log_prgx = torch.log_softmax(log_prgx, dim=1) # normalizing over runlengths

            else:
                log_pygx = log_pi_t

            if prgx is not None:
                if task_supervision is not None:
                    override_log_prgx = torch.log(prgx[i]) + torch.log(task_supervision[i].unsqueeze(-1))
                    masked_log_prgx = log_prgx + torch.log(1-task_supervision[i].unsqueeze(-1))
                    cat_log_prgx = torch.cat((override_log_prgx.unsqueeze(-1), masked_log_prgx.unsqueeze(-1)),dim=-1)
                    log_prgx = torch.logsumexp(cat_log_prgx,dim=-1)
                else:
                    log_prgx = torch.log(prgx[i])
                    
            if not return_timing: log_prgx_list.append(log_prgx)

            # use these posterior predictives and log p(r | hist) to evaluate y under the full posterior predictive
            nll = self.nll(log_pygx, log_prgx)
            if not return_timing: nll_list.append(nll)

            # update belief over run lengths:

            # if classification, then log_pi_t is (batch_size, i+1. y_dim), need to mask before updating belief
            if self.classification:
                log_pygx = mask_nlls(y.unsqueeze(-2), log_pygx) # (batch_size, i+1)

            # calculate joint densities p(r_t,data_so_far) for both r_t = r_{t-1} + 1 and r_t = 0
            log_prx_grow = self.log_1m_hazard + log_pygx + log_prgx                      # p( r_t = r_{t-1} + 1 )
            log_prx_chpt = self.log_hazard + torch.logsumexp(log_pygx + log_prgx, dim=1) # p (r_t = 0 )

            log_prx = torch.cat((log_prx_grow,log_prx_chpt[:,None]), 1) # shape (batch_size, i + 2)
            log_prgx = torch.log_softmax(log_prx, dim=1) # log p(r_{i+1} | x_{0:i}, y_{0:i})

            # update posteriors update
            posterior_params = tuple( torch.cat((u, p.expand(*([batch_size] + list(p.shape)[1:]))), axis=1) for u,p in zip(updated_posterior_params, prior_params) )

            # append to list
            if not return_timing: posterior_params_list.append(posterior_params)

            if return_timing:
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                time_array.append(end_time-start_time)
            
        if not return_timing: nlls = torch.stack(nll_list, dim=1) # shape (batch_size, t, y_dim)
        
        if return_timing:
            return [], [], [], time_array
        
        return posterior_params_list, log_prgx_list, nlls
