__author__ = 'jonatank'

import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import total_variation
from utils.ssim import ssim
from utils.utils import normalize_tensor

def run_map_TV(input_img, dec_mu, riter, device, weight = 1, step_size=0.003):
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

        # Define G function
        gfunc = (torch.sum((dec_mu-img_ano).pow(2)) + weight * total_variation(img_ano-input_img))

        gfunc.backward() # Backpropogate

        torch.clamp(img_ano, -100, 100) # Limit gradients to -100 and 100

        MAP_optimizer.step() # X' = X + lr_rate*grad(gfunc(X))
        MAP_optimizer.zero_grad()

    return img_ano

def run_map_NN(input_img, dec_mu, model, riter, device, writer=None, optimizer=None, input_seg=None , mode=None , step_size=0.003):
    # Init params
    input_img = nn.Parameter(input_img, requires_grad=False)
    dec_mu = nn.Parameter(dec_mu.float(), requires_grad=False)
    img_ano = nn.Parameter(input_img.clone(),requires_grad=True)

    input_img = input_img.to(device)
    dec_mu = dec_mu.to(device)
    img_ano = img_ano.to(device)

    if mode == 'train':
        input_seg = input_seg.to(device)

    model.eval()

    # Init MAP Optimizer
    MAP_optimizer = optim.Adam([img_ano], lr=step_size)
    ssim_loss = 0

    for i in range(riter):
        print(i)
        if mode == 'train':
            model.train()

            NN_input = torch.stack([input_img, img_ano]).permute((1,0,2,3)).float()

            # Define G function
            gfunc = torch.sum((dec_mu-img_ano).pow(2)) + torch.sum(model(NN_input))

            gfunc.backward()

            ssim_i = 1 - ssim((input_img-img_ano).pow(2).unsqueeze(1).float() , input_seg.unsqueeze(1).float())

            ssim_loss += ssim_i

            ssim_i.backward()
            optimizer.step()
            optimizer.zero_grad()

            model.eval()

        NN_input = torch.stack([input_img, img_ano]).permute((1,0,2,3)).float()

         # Define G function
        gfunc = torch.sum((dec_mu-img_ano).pow(2)) + torch.sum(model(NN_input))

        gfunc.backward()

        MAP_optimizer.step()
        MAP_optimizer.zero_grad()


    writer.add_scalar('ssim:', ssim_loss/riter)
    writer.flush()

    return img_ano
