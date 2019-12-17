import torch
import torch.nn as nn
from metacpd.main.encoders import get_encoder
from metacpd.main.utils import Flatten, conv_block, final_conv_block

class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        hid_dim = config['model.hid_dim']
        self.use_cuda = config['data.cuda']
        self.y_dim = config['model.y_dim']
        self.phi_dim = config['model.phi_dim']

        self.config = config
        encoder = get_encoder(config)
        self.encoder = nn.Sequential(
            encoder,
            nn.Linear(self.phi_dim, self.y_dim),
            nn.LogSoftmax()
        )

    def forward(self,x,y,prgx=None, task_supervision=None):
        """
        x: (batch_size, horizon, C, H, W)
        y: (batch_size, horizon, n_classes)

        output: nlls (batch_size, horizon)
        """
        #reshape x_mat
        x_shape = x.shape

        nlls = -self.encoder(x.reshape(-1,x_shape[-3],x_shape[-2],x_shape[-1]))
        nlls = nlls.reshape(x_shape[0], x_shape[1], self.y_dim)
        
        return None, None, nlls
