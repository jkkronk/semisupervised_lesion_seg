__author__ = 'jonatank'

import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import total_variation
from utils.ssim import ssim
from utils.utils import normalize_tensor
from utils.utils import dice_loss, diceloss

def run_map_TV(input_img, dec_mu, vae_model, riter, device, weight = 1, step_size=0.003):
    # Init params
    input_img = nn.Parameter(input_img, requires_grad=False)
    dec_mu = nn.Parameter(dec_mu.float(), requires_grad=False)
    img_ano = nn.Parameter(input_img.clone(),requires_grad=True)

    input_img = input_img.to(device)
    dec_mu = dec_mu.to(device)
    img_ano = img_ano.to(device)

    # Init Optimizer
    MAP_optimizer = optim.Adam([img_ano], lr=step_size)

    # Iterate until convergence
    for i in range(riter):
        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        # Define G function
        l2_loss = (dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2)
        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())

        gfunc = torch.sum(l2_loss) + kl_loss + weight * total_variation(img_ano-input_img)

        gfunc.backward() # Backpropogate

        torch.clamp(img_ano, -100, 100) # Limit gradients to -100 and 100

        MAP_optimizer.step() # X' = X + lr_rate*grad(gfunc(X))
        MAP_optimizer.zero_grad()

    return img_ano

def run_map_NN(input_img, dec_mu, net, vae_model, riter, device, writer=None, step_size=0.003, input_seg=None):
    # Init params
    input_img = nn.Parameter(input_img.to(device), requires_grad=False)
    dec_mu = nn.Parameter(dec_mu.to(device).float(), requires_grad=False)
    img_ano = nn.Parameter(input_img.clone().to(device),requires_grad=True)

    # Init MAP Optimizer
    MAP_optimizer = optim.Adam([img_ano], lr=step_size)

    net.eval()

    for i in range(riter):
        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        # Define G function
        l2_loss = (dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2)
        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())

        gfunc = torch.sum(l2_loss) + kl_loss #+ torch.sum(out)

        gfunc.backward() # Backpropagate

        NN_input = torch.stack([input_img, img_ano, img_ano.grad]).permute((1, 0, 2, 3)).to(device).float()
        out = net(NN_input)

        img_ano.grad = out.squeeze(1).double()

        MAP_optimizer.step() # Gradient step

        if i % 25 == 0 and not writer == None: # Log
            writer.add_scalar('ELBO grad', torch.sum(l2_loss) + kl_loss)
            writer.add_scalar('Net grad', torch.sum(net(NN_input)))
            #writer.add_image('X_i ano_img', normalize_tensor(img_ano.unsqueeze(1)[:16]), dataformats='NCHW')
            writer.add_image('X_i ano_img_grad', normalize_tensor(img_ano.grad.unsqueeze(1)[:16]), dataformats='NCHW')
            writer.add_image('X_i ano_img_grad', normalize_tensor(out[:16]), dataformats='NCHW')

            if not input_seg == None:
                input_seg = input_seg.to(device)
                dice = diceloss()
                loss = dice(img_ano.grad, input_seg)
                writer.add_scalar('Iteration loss valid', loss)

        MAP_optimizer.zero_grad()
    return img_ano

def train_run_map_NN(input_img, dec_mu, net, vae_model, riter, device, writer=None, optimizer=None, input_seg=None, step_size=0.003, log_freq=5):
    # Init params
    input_img = nn.Parameter(input_img, requires_grad=False)
    dec_mu = nn.Parameter(dec_mu.to(device).float(), requires_grad=False)
    img_ano = nn.Parameter(input_img.clone().to(device),requires_grad=True)
    input_seg = input_seg.to(device)

    # Init MAP Optimizer
    MAP_optimizer = optim.Adam([img_ano], lr=step_size)

    net.train()
    optimizer.zero_grad()

    dice = diceloss()
    #BCE = nn.BCELoss()

    for i in range(riter):
        MAP_optimizer.zero_grad()

        # Gradient step MAP
        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        # Define G function
        l2_loss = (dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2)
        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())

        gfunc = torch.sum(l2_loss) + kl_loss #+ torch.sum(out)
        gfunc.backward()

        NN_input = torch.stack([input_img, img_ano]).permute((1, 0, 2, 3)).float()
        out = net(NN_input)

        img_ano.grad += out.squeeze(1)

        ano_grad_act = 1 - 2 * torch.sigmoid(-1 * img_ano.grad.pow(2))

        loss = dice(ano_grad_act, input_seg)
        # loss = BCE(ano_grad_act.double(), input_seg.double())
        # loss = 1 - ssim(ano_grad_act.unsqueeze(1).float(), input_seg.unsqueeze(1).float())

        loss.backward()

        # Update Img_ano
        MAP_optimizer.step()

        if i % log_freq == 0: # Log
            writer.add_scalar('ELBO grad', torch.sum(l2_loss) + kl_loss)
            writer.add_scalar('Net grad', torch.sum(net(NN_input)))

    writer.add_scalar('Iteration loss', loss)
    # Update network parameters
    optimizer.step()

    # Log
    writer.add_image('X Img', normalize_tensor(input_img.unsqueeze(1)[:16]), dataformats='NCHW')
    writer.add_image('X Seg', normalize_tensor(input_seg.unsqueeze(1)[:16]), dataformats='NCHW')
    writer.add_image('X Ano_grad_act', normalize_tensor(ano_grad_act.unsqueeze(1)[:16]), dataformats='NCHW')
    writer.add_image('X_i ano_img', normalize_tensor(img_ano.unsqueeze(1)[:16]), dataformats='NCHW')
    writer.add_image('X_i out', normalize_tensor(out[:16]), dataformats='NCHW')
    writer.add_image('X_i ano_img_grad', normalize_tensor(img_ano.grad.unsqueeze(1)[:16]), dataformats='NCHW')

    return img_ano