__author__ = 'jonatank'

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from networks.vae_bilinear_conv import encoder_layer, latent_layer_1, latent_layer_2, decoder_layer

class ConvVAE(nn.Module):
    def __init__(self, img_size, name):
        super(ConvVAE, self).__init__()
        # Parameters
        self.img_size = img_size
        self.input_size = 1 # 1 channel (B/W) for RGB set 3
        self.name = name
        self.gf_dim = 16

        ### Encoder network
        self.encoder, self.res_encoder = encoder_layer(self.input_size, self.gf_dim)

        # hidden => mu
        self.fc1 = latent_layer_1(self.gf_dim)

        # hidden => logvar
        self.fc2 = latent_layer_2(self.gf_dim)

        ### Decoder network
        self.decoder = decoder_layer(self.gf_dim)

    def encode(self, x):
        # Encoder layers
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        res = self.res_encoder(x)
        return mu, logvar, res

    def decode(self, z):
        # Deconder layers
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        # Repametrization trick to make backprop. possible
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar, res = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, res

    def sample(self, batch_size, device):
        # Saple latent variables z
        # latent_size = self.gf_dim*4 # Change when other latent layer
        sample = torch.randn(batch_size,512,2,2, dtype=torch.double).to(device) #batch_size, latent_size)
        return self.decode(sample)

def loss_function(recon_x, x, res, mu, logvar, weight = 1):
    # Autoencoder loss - reconstruction loss
    l2_loss = torch.sum((recon_x.view(-1, recon_x.numel()) - x.view(-1, x.numel())).pow(2)) # * 0.5

    # Latent loss
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #0.5 * torch.sum(mu.pow(2) + logvar.pow(2) - 2.*torch.log(torch.abs(logvar)) - 1)
    # Residual loss
    true_residuals = torch.abs(x-recon_x)
    autoencoder_res_loss = torch.sum((res - true_residuals).pow(2))

    # Total loss sum of all
    return torch.sum(l2_loss + weight*kl_divergence_loss + autoencoder_res_loss), kl_divergence_loss, l2_loss, autoencoder_res_loss

def train_vae(model, train_loader, device, optimizer, epoch):
    # Params
    model.train()
    train_loss = 0
    train_lat_loss = 0
    train_gen_loss = 0
    train_res_loss = 0

    weight = 1#(epoch%25+1)/25 #if epoch < 50 else 1

    for batch_idx, (data, _) in enumerate(train_loader): #tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar, res = model(data.double())
        loss, lat_loss, gen_loss, res_loss  = loss_function(recon_batch, data, res, mu, logvar, weight)

        train_loss += loss.item()
        train_lat_loss += lat_loss.item()
        train_gen_loss += gen_loss.item()
        train_gen_loss += res_loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_lat_loss /= len(train_loader.dataset)
    train_gen_loss /= len(train_loader.dataset)
    train_res_loss /= len(train_loader.dataset)

    return train_loss, train_lat_loss, train_gen_loss, train_res_loss


def valid_vae(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    test_lat_loss = 0
    test_gen_loss = 0
    test_res_loss = 0

    weight = 1#(epoch%25+1)/25 #epoch/50 if epoch < 50 else 1

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader): #tqdm(enumerate(test_loader), total=len(test_loader), desc='test'):
            data = data.to(device)

            recon_batch, mu, logvar, res = model(data.double())
            loss, lat_loss, gen_loss, res_loss = loss_function(recon_batch, data, res, mu, logvar, weight)

            test_loss += loss.item()
            test_lat_loss += lat_loss.item()
            test_gen_loss += gen_loss.item()
            test_res_loss += res_loss.item()

    test_loss /= len(test_loader.dataset)
    test_lat_loss /= len(test_loader.dataset)
    test_gen_loss /= len(test_loader.dataset)
    test_res_loss /= len(test_loader.dataset)

    return test_loss, test_lat_loss, test_gen_loss, test_res_loss

def plot_restored(path, img_batch, batch_size,img_nbr = 0, img_size=128):
    plt.imsave(path,img_batch.view(batch_size, 1, img_size, img_size)[img_nbr,0].detach().numpy())

