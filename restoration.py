__author__ = 'jonatank'

import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import total_variation
from utils.ssim import ssim

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

    model.eval()

    # Init MAP Optimizer
    MAP_optimizer = optim.Adam([img_ano], lr=step_size)

    for i in range(riter):
        if mode == 'train':
            model.train()
            input_seg = input_seg.to(device)

            #img_ano.requires_grad=False

            NN_input = torch.stack([input_img, img_ano]).permute((1,0,2,3)).float()

            # Define G function
            gfunc = torch.sum((dec_mu-img_ano).pow(2)) + torch.sum(model(NN_input))

            gfunc.backward()

            #print(((dec_mu-img_ano.grad).pow(2).unsqueeze(1) + weight * pred).size(), input_seg.size())
            print(input_seg.size())
            print(img_ano.grad.size())

            ssim_i = 1 - ssim(img_ano.grad.unsqueeze(1).float() , input_seg.unsqueeze(1).float())

            print(ssim_i)

            ssim_i.backward()
            optimizer.step()
            optimizer.zero_grad()

            model.eval()
        else:
            print(i)

        NN_input = torch.stack([input_img, img_ano]).permute((1,0,2,3)).float()

         # Define G function
        gfunc = torch.sum((dec_mu-img_ano).pow(2)) + torch.sum(model(NN_input))

        gfunc.backward()

        MAP_optimizer.step()
        MAP_optimizer.zero_grad()

        #writer.add_image('input_seg', input_seg.unsqueeze(1), dataformats='NCHW')
        #writer.add_image('input-ano', (input_img - img_ano).pow(2).unsqueeze(1), dataformats='NCHW')
        #writer.flush()
        #exit()
        #print(input_img.size(), img_ano.size(), input_seg.size())

        seg_ssim = ssim(((input_img - img_ano).pow(2)).unsqueeze(1).float(), input_seg.unsqueeze(1)) #, data_range=1, size_average=False)[0]
        print(i, seg_ssim)

    return img_ano
