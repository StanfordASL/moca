import torch
import torch.nn as nn
from metacpd.main.utils import Flatten, conv_block, final_conv_block

def get_encoder(config):
    hid_dim = config['model.hid_dim']
    x_dim = config['model.x_dim']
    phi_dim = config['model.phi_dim']
    # REGRESSION

    if config['data.dataset'] == 'Sinusoid':
        activation = nn.Tanh()
        encoder = nn.Sequential(
            nn.Linear(x_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, phi_dim),
            activation
        )
    elif config['data.dataset'] == 'NoiseSinusoid':
        activation = nn.Tanh()
        encoder = nn.Sequential(
            nn.Linear(x_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, phi_dim),
            activation
        )
        
    # CLASSIFICATION

    elif config['data.dataset'] in ['RainbowMNIST']:
        encoder = nn.Sequential(
            conv_block(3, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            final_conv_block(hid_dim, hid_dim),
            Flatten()
        )
    elif config['data.dataset'] == 'MiniImageNet':
        encoder = nn.Sequential(
            conv_block(3, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            final_conv_block(hid_dim, hid_dim),
            Flatten()
        )

    elif config['data.dataset'] == 'PermutedMNIST':
        encoder = nn.Sequential(
            conv_block(1, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            final_conv_block(hid_dim, hid_dim),
            Flatten()
        )

    else:
        raise ValueError("data.dataset not understood")

    return encoder
