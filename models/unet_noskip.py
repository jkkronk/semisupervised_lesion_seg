import torch
import torch.nn as nn
from collections import OrderedDict

# FROM GITHUB https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py

class UNET_noskip(nn.Module):
    def __init__(self, name,in_channels=1, out_channels=1, init_features=32):
        super(UNET_noskip, self).__init__()
        self.name = name

        features = init_features
        self.encoder1 = UNET_noskip._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNET_noskip._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNET_noskip._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNET_noskip._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNET_noskip._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNET_noskip._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNET_noskip._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNET_noskip._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNET_noskip._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class MSELoss(nn.Module):
    def forward(self, input, target):
        target_rec = torch.sigmoid(input)
        loss = torch.mean((target-target_rec)**2)

        return loss

def train_unet_noskip(model, train_loader, device, optimizer):
    # Params
    model.train()
    train_loss = 0
    criterion = MSELoss()
    for batch_idx, (scan, mask) in enumerate(train_loader): #tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
        scan = scan.to(device)

        optimizer.zero_grad()
        pred = model(scan.float())
        loss = criterion(pred, scan)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)

    return train_loss

def valid_unet_noskip(model, test_loader, device):
    # Params
    model.eval()
    valid_loss = 0
    criterion = MSELoss()
    for batch_idx, (scan, mask) in enumerate(test_loader): #tqdm(enumerate(test_loader), total=len(test_loader), desc='validation'):
        scan = scan.to(device)

        pred = model(scan.float())
        loss = criterion(pred, scan)
        valid_loss += loss.item()

    valid_loss /= len(test_loader.dataset)

    return valid_loss

