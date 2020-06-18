import torch.nn as nn
import torch.nn.functional as F

class ResBlock_Down(nn.Module):
    """
    Res block for convolution.
    1) One Conv2d() layer and a following Conv2d(stride=1) with in_layer between.
    2) One a Conv2d() layer with input and output directly.
    Output is these two summarized
    """
    def __init__(self, input_size, out_layers, stride=2, padding=1, act=True):
        super(ResBlock_Down, self).__init__()
        self.input_size = input_size
        self.in_layers = input_size
        self.out_layers = out_layers
        self.stride = stride
        self.pad = padding
        self.act = act

        self.conv1 = nn.Conv2d(self.input_size, self.in_layers, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_layers)
        self.conv2 = nn.Conv2d(self.in_layers, self.out_layers, kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_layers)

        if self.act:
            self.shortcut = nn.Sequential(
            nn.Conv2d(self.input_size, self.out_layers, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_layers),
            nn.LeakyReLU(0.2)
            )
        else:
            self.shortcut = nn.Sequential(
            nn.Conv2d(self.input_size, self.out_layers, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_layers)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        if self.act:
            out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2)
        else:
            out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        return out

class ResBlock_Up(nn.Module):
    """
    Res block for transpose convolution.
    1) One transpose Conv2d() layer and a following transpose Conv2d(stride=1) with in_layer between.
    2) One a transpose Conv2d() layer with input and output directly.
    Output is these two summarized
    """
    def __init__(self, input_size, out_layers, padding=1):
        super(ResBlock_Up, self).__init__()
        self.input_size = input_size
        self.in_layers = input_size
        self.out_layers = out_layers
        self.pad = padding

        #self.conv1 = nn.ConvTranspose2d(self.input_size, self.in_layers, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.input_size, self.in_layers, kernel_size=3, stride=2, padding=1)
        )
        self.bn1 = nn.BatchNorm2d(self.in_layers)
        #self.conv2 = nn.ConvTranspose2d(self.in_layers, self.out_layers, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.in_layers, self.out_layers, kernel_size=3, stride=1, padding=1)
        )
        self.bn2 = nn.BatchNorm2d(self.out_layers)

        self.shortcut = nn.Sequential(
            #nn.ConvTranspose2d(self.input_size, self.out_layers, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(self.input_size, self.out_layers, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_layers)
        )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2)
        out += F.leaky_relu(self.shortcut(x), 0.2)
        return out

def encoder_layer(input_size, gf_dim):
    """
    Encoder function returning encoder layer
    """

    fist_encoder_layer = nn.Sequential(
        nn.Conv2d(input_size, gf_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(gf_dim),
        nn.LeakyReLU(0.2)
    )

    encoder = nn.Sequential(
        fist_encoder_layer,
        ResBlock_Down(gf_dim, gf_dim),
        ResBlock_Down(gf_dim, gf_dim*2),
        ResBlock_Down(gf_dim*2, gf_dim*4),
        ResBlock_Down(gf_dim*4, gf_dim*8),
        ResBlock_Down(gf_dim*8, gf_dim*16)
    )

    res_encoder = nn.Sequential(
        fist_encoder_layer,
        nn.Conv2d(gf_dim, gf_dim, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(gf_dim),
        nn.LeakyReLU(0.2),
        nn.Conv2d(gf_dim, gf_dim*2, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(gf_dim*2),
        nn.LeakyReLU(0.2),
        nn.Conv2d(gf_dim*2, gf_dim, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(gf_dim),
        nn.LeakyReLU(0.2),
        nn.Conv2d(gf_dim, 1, kernel_size=3, padding=2, dilation=2)
    )

    return encoder, res_encoder

def latent_layer_1(gf_dim):
    """
    Function used for returning mu
    """
    return ResBlock_Down(gf_dim*16, gf_dim*32, act=False)

def latent_layer_2(gf_dim):
    """
    Function used for returning std
    """
    return ResBlock_Down(gf_dim*16, gf_dim*32, act=False)

def decoder_layer(gf_dim):
    """
    Decoder function returning decoder layer
    """
    decoder = nn.Sequential(
        ResBlock_Up(gf_dim*32, gf_dim*16),
        ResBlock_Up(gf_dim*16, gf_dim*8),
        ResBlock_Up(gf_dim*8, gf_dim*4),
        ResBlock_Up(gf_dim*4, gf_dim*2),
        ResBlock_Up(gf_dim*2, gf_dim),
        ResBlock_Up(gf_dim, gf_dim),
        nn.Conv2d(gf_dim, gf_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(gf_dim),
        nn.LeakyReLU(0.2),
        nn.Conv2d(gf_dim, 1, kernel_size=3, stride=1, padding=1)
        )
    return decoder