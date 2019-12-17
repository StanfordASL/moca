import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from metacpd.main.encoders import get_encoder

class PCOC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = deepcopy(config)
        self.x_dim = config['model.x_dim']
        self.phi_dim = config['model.phi_dim']
        self.y_dim = config['model.y_dim']

        self.sigma_eps = np.zeros([self.y_dim,1]) + np.asarray(eval(config['model.sigma_eps']))
        self.cov_dim =  self.sigma_eps.shape[-1]
        print("Using %d parameters in covariance:" % self.cov_dim)
        if self.phi_dim % self.cov_dim != 0:
            raise ValueError("cov_dim must evenly divide phi_dim")

        self.logSigEps = nn.Parameter(torch.from_numpy(np.log(self.sigma_eps)), requires_grad=self.config['train.learnable_noise'])
        
        Linv_offset = config['model.Linv_init']
        dir_scale = config['model.dirichlet_scale']
        self.Q = nn.Parameter(torch.randn(self.y_dim, self.cov_dim, self.phi_dim//self.cov_dim))
        self.logLinv = nn.Parameter(torch.randn(self.y_dim, self.cov_dim)+Linv_offset)
        self.log_dirichlet_priors = nn.Parameter(dir_scale*torch.ones(self.y_dim), requires_grad=config['train.learnable_dirichlet'])

        self.normal_nll_const = self.phi_dim*np.log(2*np.pi)

        self.encoder = get_encoder(config)

    @property
    def invSigEps(self):
        return torch.exp(-self.logSigEps) #.repeat(self.y_dim,1)

    @property
    def SigEps(self):
        return torch.exp(self.logSigEps) #.repeat(self.y_dim,1)

    def prior_params(self):
        Q0 = self.Q
        Linv0 = torch.exp(self.logLinv)
        dir_weights = torch.exp(self.log_dirichlet_priors)

        return (Q0, Linv0, dir_weights)

    def recursive_update(self, phi, y, params):
        """
            inputs: phi: shape (..., cov_dim, k )
                    y:   shape (..., y_dim )
                    params: tuple of Q, Linv
                        Q: shape (..., y_dim, cov_dim, k)
                        Linv: shape (..., y_dim, cov_dim)
                        dir_weights: shape (..., y_dim)
        """
        Q, Linv, dir_weights = params

        # zeros out entries all except class y
        invSigEps_masked = self.invSigEps * y.unsqueeze(-1) # (..., y_dim, cov_dim)

        Q = Q + invSigEps_masked.unsqueeze(-1)*phi.unsqueeze(-3)
        Linv = Linv + invSigEps_masked
        dir_weights = dir_weights + y

        return (Q, Linv, dir_weights)

    def log_predictive_prob(self, x, y, posterior_params, update_params=False):
        """
            input:  x: shape (..., x_dim)
                    y: shape (..., y_dim)
                    posterior_params: tuple of Q, Linv:
                        Q: shape (..., y_dim, cov_dim, k)
                        Linv: shape (..., y_dim, cov_dim)
                        dir_weights: shape (..., y_dim)
                    update_params: bool, whether to perform recursive update on
                                   posterior params and return updated params
            output: logp: log p(y, x | posterior_params) (..., y_dim)
                    updated_params: updated posterior params after factoring in (x,y) pair
        """

        x_shape = list(x.shape)

        if len(x_shape) > 4: # more than one batch dim
            x = x.reshape([-1]+x_shape[-3:])

        phi = self.encoder(x) # (..., phi_dim)
        if len(x_shape) > 4:
            phi = phi.reshape(x_shape[:-3]+[self.phi_dim])

        Q, Linv, dir_weights = posterior_params
        mu = Q / Linv.unsqueeze(-1) # (..., y_dim, cov_dim, k)
        pred_cov = 1./Linv + self.SigEps() # (..., y_dim, cov_dim)

        phi_shape = phi.shape
        phi_reshaped = phi.reshape(*(list(phi_shape)[:-1]+[self.cov_dim, -1])) # (..., cov_dim, k)

        err = phi_reshaped.unsqueeze(-3) - mu # (..., y_dim, cov_dim, k)

        nll_quadform = (err**2 / pred_cov.unsqueeze(-1) ).sum(-1).sum(-1)
        nll_logdet = (self.phi_dim/self.cov_dim) * torch.log(pred_cov).sum(-1) # sum of log of diagonal entries

        logp = -0.5*(nll_quadform + nll_logdet + self.normal_nll_const) # log p(x | y)

        logp += torch.log(dir_weights / dir_weights.sum(-1,keepdim=True)) # multiply by p(y) posterior to get p(x, y)

        if update_params:
            updated_params = self.recursive_update(phi_reshaped, y, posterior_params)
            return logp, updated_params

        return logp


    def forward(self, x, posterior_params):
        """
            input: x, posterior params
            output: log p(x | y) for all y
        """
        x_shape = list(x.shape)

        if len(x_shape) > 4: # more than one batch dim
            x = x.reshape([-1]+x_shape[-3:])

        phi = self.encoder(x) # (..., phi_dim)
        if len(x_shape) > 4:
            phi = phi.reshape(x_shape[:-3]+[self.phi_dim])

        Q, Linv, dir_weights = posterior_params
        mu = Q / Linv.unsqueeze(-1) # (..., y_dim, cov_dim, k)
        pred_cov = 1./Linv + self.SigEps() # (..., y_dim, cov_dim)

        phi_shape = phi.shape
        phi_reshaped = phi.reshape(*(list(phi_shape)[:-1]+[self.cov_dim, -1])) # (..., cov_dim, k)

        err = phi_reshaped.unsqueeze(-3) - mu # (..., y_dim, cov_dim, k)

        nll_quadform = (err**2 / pred_cov.unsqueeze(-1) ).sum(-1).sum(-1)
        nll_logdet = (self.phi_dim/self.cov_dim) * torch.log(pred_cov).sum(-1) # sum of log of diagonal entries

        logp = -0.5*(nll_quadform + nll_logdet + self.normal_nll_const) # log p(x | y)

        logp += torch.log(dir_weights / dir_weights.sum(-1,keepdim=True)) # multiply by p(y) posterior to get p(x, y)

        return logp
