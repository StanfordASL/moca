import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from metacpd.main.encoders import get_encoder

class ALPaCA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = deepcopy(config)
        self.x_dim = config['model.x_dim']
        self.phi_dim = config['model.phi_dim']
        self.y_dim = config['model.y_dim']

        self.sigma_eps = eval(self.config['model.sigma_eps'])
        self.logSigEps = nn.Parameter(torch.from_numpy(np.log(self.sigma_eps)), requires_grad=self.config['train.learnable_noise'])

        self.Q = nn.Parameter(torch.randn(self.phi_dim, self.y_dim))
        self.L_asym = nn.Parameter(torch.randn(self.phi_dim, self.phi_dim))

        self.normal_nll_const = self.y_dim*np.log(2*np.pi)

        hid_dim = config['model.hid_dim']
        self.encoder = get_encoder(config)

    @property
    def logdetSigEps(self):
        return torch.sum(self.logSigEps)

    @property
    def invSigEps(self):
        return torch.diag(torch.exp(-self.logSigEps))

    def prior_params(self):
        Q0 = self.Q
        Linv0 = self.L_asym @ self.L_asym.T

        return (Q0, Linv0)

    def recursive_update(self, phi, y, params):
        """
            inputs: phi: shape (..., phi_dim )
                    y:   shape (..., y_dim )
                    params: tuple of Q, Linv
                        Q: shape (..., phi_dim, y_dim)
                        Linv: shape (..., phi_dim, phi_dim)
        """
        Q, Linv = params

        Lphi = Linv @ phi.unsqueeze(-1)

        Linv = Linv - 1./(1 + phi.unsqueeze(-2) @ Lphi) * (Lphi @ Lphi.transpose(-1,-2))
        Q = phi.unsqueeze(-1) @ y.unsqueeze(-2) + Q

        return (Q, Linv)

    def log_predictive_prob(self, x, y, posterior_params, update_params=False):
        """
            input:  x: shape (..., x_dim)
                    y: shape (..., y_dim)
                    posterior_params: tuple of Q, Linv:
                        Q: shape (..., phi_dim, y_dim)
                        Linv: shape (..., phi_dim, phi_dim)
                    update_params: bool, whether to perform recursive update on
                                   posterior params and return updated params
            output: logp: log p(y | x, posterior_parms)
                    updated_params: updated posterior params after factoring in (x,y) pair
        """

        phi = self.encoder(x)

        Q, Linv = posterior_params

        K = Linv @ Q

        sigfactor = 1 + (phi.unsqueeze(-2) @ Linv @ phi.unsqueeze(-1))
        err = y.unsqueeze(-1) - K.transpose(-1,-2) @ phi.unsqueeze(-1)

        invsig = self.invSigEps / sigfactor # shape (..., y_dim y_dim)

        nll_quadform = err.transpose(-1,-2) @ invsig @ err
        nll_logdet = self.y_dim * torch.log(sigfactor) + self.logdetSigEps

        logp = -0.5*(self.normal_nll_const + nll_quadform + nll_logdet).squeeze(-1).squeeze(-1)

        if update_params:
            updated_params = self.recursive_update(phi,y,posterior_params)
            return logp, updated_params

        return logp


    def forward(self, x, posterior_params):
        """
            input: x, posterior params
            output: y
        """
        phi = self.encoder(x)

        Q, Linv = posterior_params

        K = Linv @ Q

        sigfactor = 1 + (phi.unsqueeze(-2) @ Linv @ phi.unsqueeze(-1))
        mu = ( K.transpose(-1,-2) @ phi.unsqueeze(-1) ).squeeze(-1)
        invsig = self.invSigEps / sigfactor

        return mu, invsig
