import torch
import torch.nn as nn

from tqdm import tqdm
from collections import OrderedDict

class ConvNet(nn.Module):
    def __init__(self, name, img_ano, vae_model, dec_mu):
        super(ConvNet, self).__init__()
        self.name = name
        self.img = img_ano
        self.prior = dec_mu
        self.vae = vae_model

        # Gradient step MAP
        __, z_mean, z_cov, __ = self.vae(self.img.unsqueeze(1).double())

        # Define G function
        l2_loss = (self.prior.view(-1, self.prior.numel()) - self.img.view(-1, self.img.numel())).pow(2)
        self.kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())


    def forward(self, x):


        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        return out
