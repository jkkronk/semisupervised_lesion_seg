import torch
import torch.nn as nn

from tqdm import tqdm
from collections import OrderedDict

class ConvNet(nn.Module):
    def __init__(self, name,in_channels=2, out_channels=1, init_features=16):
        super(ConvNet, self).__init__()
        self.name = name

        #2x128x128
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, init_features, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #init_featuresx64x64
        self.layer2 = nn.Sequential(
            nn.Conv2d(init_features, init_features*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #init_features*2x32x32
        self.layer3 = nn.Sequential(
            nn.Conv2d(init_features*2, init_features*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #init_features*4x16x16
        #self.layer4 = nn.Sequential(
        #    nn.Conv2d(init_features * 4, init_features*8, kernel_size=3, stride=1, padding=1),
        #    nn.LeakyReLU(),
        #    nn.MaxPool2d(kernel_size=2, stride=2))
        #init_features*8x8x8
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(16 * 16 * init_features*4, out_channels)

        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        return out
