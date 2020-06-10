__author__ = 'jonatank'

import torch
import torch.nn as nn
from utils.ssim import ssim
from utils.utils import normalize_tensor, normalize_tensor_N
from utils.utils import diceloss

def run_map_NN(input_img, mask, dec_mu, net, vae_model, riter, device, input_seg=None, threshold=None, writer=None, step_size=0.003, log=True):
    # Init params
    input_img = input_img.to(device)
    mask = mask.to(device)
    dec_mu = dec_mu.to(device).float()
    img_ano = nn.Parameter(input_img.clone().to(device), requires_grad=True)

    net.eval()
    for i in range(riter):
        # Gradient function
        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())
        l2_loss = torch.sum((dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2))

        gfunc = l2_loss + kl_loss

        ELBO_grad, = torch.autograd.grad(gfunc, img_ano,
                                    grad_outputs=gfunc.data.new(gfunc.shape).fill_(1),
                                    create_graph=True)

        NN_input = torch.stack([input_img, img_ano.detach()]).permute((1, 0, 2, 3)).float()
        out = net(NN_input).squeeze(1)
        img_grad = ELBO_grad.detach() * out

        img_ano_update = img_ano - step_size * img_grad * mask

        img_ano = img_ano_update.detach()
        img_ano.requires_grad = True

    # Log
    if log and not writer == None :
        writer.add_image('Img', normalize_tensor(input_img.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Seg', normalize_tensor(input_seg.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Restored', normalize_tensor(img_ano.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Restored_Img', normalize_tensor((img_ano - input_img).pow(2).unsqueeze(1)[:16]),
                         dataformats='NCHW')
        writer.add_image('Out', normalize_tensor(out.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('ELBO', normalize_tensor(ELBO_grad.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Grad', normalize_tensor(img_grad.unsqueeze(1)[:16]), dataformats='NCHW')

        resultSeg = torch.abs(img_ano - input_img)
        resultSeg[resultSeg >= threshold] = 1
        resultSeg[resultSeg < threshold] = 0

        writer.add_image('ResultSeg', normalize_tensor(resultSeg.unsqueeze(1)[:16]), dataformats='NCHW')
        #if torch.sum(out.flatten()) > 0 and torch.sum(grad.flatten()) > 0:
        #    writer.add_histogram('hist-out-torch', out.flatten())
        #    writer.add_histogram('hist-grad-torch', grad.flatten())

        writer.flush()

    return img_ano

def train_run_map_NN(input_img, dec_mu, net, vae_model, riter, K_actf, step_size, device, writer, input_seg, mask,
                     train=True,log=True):
    # Init params
    input_img = input_img.to(device)
    mask = mask.to(device)
    dec_mu = dec_mu.to(device).float()
    input_seg = input_seg.to(device)
    img_ano = nn.Parameter(input_img.clone().to(device),requires_grad=True)

    if train:
        net.train()

    # Init MAP Optimizer
    #criterion = diceloss()
    criterion = nn.BCELoss()
    tot_loss = 0
    for i in range(riter):
        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())
        l2_loss = torch.sum((dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2))

        gfunc = l2_loss + kl_loss

        ELBO_grad, = torch.autograd.grad(gfunc, img_ano, grad_outputs=gfunc.data.new(gfunc.shape).fill_(1),
                                         create_graph=True)

        NN_input = torch.stack([input_img, img_ano.detach()]).permute((1, 0, 2, 3)).float()
        out = net(NN_input).squeeze(1)
        img_grad = ELBO_grad.detach() * out

        img_ano_update = img_ano.detach() - step_size * img_grad * mask

        # img_ano_act = 1 - 2 * torch.sigmoid(-K_actf*(img_ano_update - input_img).pow(2))
        # img_ano_act = torch.tanh(normalize_tensor_N((img_ano_update - input_img).pow(2), K_actf))
        img_ano_act = torch.tanh(K_actf*(img_ano_update - input_img).pow(2))

        #loss = criterion(img_ano_act, input_seg)
        loss = criterion(img_ano_act.double(), input_seg.double())
        #loss = 1 - ssim(img_ano_act.unsqueeze(1).float(), input_seg.unsqueeze(1).float())

        loss.backward()

        img_ano = img_ano_update.detach()
        img_ano.requires_grad = True

        #for name, param in net.named_parameters():
        #    if param.requires_grad:
        #        print(param.grad)

        tot_loss += loss.item()

    #img_ano_act = torch.tanh(K_actf * (img_ano - input_img).pow(2))
    #loss = dice(img_ano_act, input_seg)
    #loss.backward()  # retain_graph=True)
    #tot_loss += loss.item()

    # Log
    if log:
        writer.add_image('Img', normalize_tensor(input_img.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Seg', normalize_tensor(input_seg.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Restored', normalize_tensor(img_ano.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Restored_Img', normalize_tensor((img_ano-input_img).pow(2).unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Restored_Img_act', normalize_tensor(img_ano_act.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Out', normalize_tensor(out.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('ELBO', normalize_tensor(ELBO_grad.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('Grad', normalize_tensor(img_grad.unsqueeze(1)[:16]), dataformats='NCHW')
        #writer.add_histogram('hist-out-torch', out.flatten())
        #writer.add_histogram('hist-grad-torch', grad.flatten())
        #writer.add_histogram('hist-out-mask-torch', out[input_seg > 0].flatten())
        #writer.add_histogram('hist-grad-mask-torch', grad[input_seg > 0].flatten())
        writer.flush()

    #del grad
    #del out
    #del input_img
    #del input_seg
    #del img_ano_act
    #del img_ano_update

    return img_ano, tot_loss/riter

def train_run_map_NN_CNN(input_img, dec_mu, net, vae_model, riter, device, writer, input_seg, mask,
                     optimizer=None, step_size=0.003, train=True, log=True):
    # Init params
    input_img = input_img.to(device)
    mask = mask.to(device)
    dec_mu = dec_mu.to(device).float()
    input_seg = input_seg.to(device)
    img_ano = nn.Parameter(input_img.clone().to(device),requires_grad=True)

    if train:
        net.train()
        optimizer.zero_grad()

    # Init MAP Optimizer
    dice = diceloss()
    tot_loss = 0

    for i in range(riter):
        print(i)
        img_ano.requires_grad = True

        #img_ano[mask == 0].requires_grad = False

        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())
        l2_loss = torch.sum((dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2))

        NN_input = torch.stack([input_img, img_ano]).permute((1, 0, 2, 3)).float()
        gfunc = l2_loss + kl_loss + net(NN_input)

        grad, = torch.autograd.grad(gfunc, img_ano, grad_outputs=gfunc.data.new(gfunc.shape).fill_(1),
                                    create_graph=True)

        #NN_input = torch.stack([input_img, img_ano]).permute((1, 0, 2, 3)).float()
        #out = net(NN_input).squeeze(1)
        img_grad = grad# * out

        img_ano_update = torch.zeros(img_ano.detach().shape).to(device).double()

        img_ano_update[mask > 0] = img_ano[mask > 0] - step_size * img_grad[mask > 0]

        img_ano = img_ano_update.detach()

        img_ano_act = torch.tanh(1000*(img_ano_update - input_img).pow(2))

        loss = dice(img_ano_act, input_seg)
        #loss = nn.BCELoss(ano_grad_act.double(), input_seg.double())
        #loss = 1 - ssim(ano_grad_act.unsqueeze(1).float(), input_seg.unsqueeze(1).float())
        loss.backward() #retain_graph=True

        #for name, param in net.named_parameters():
        #    if param.requires_grad:
        #        print(param.grad)

        tot_loss += loss.item()

    if train:
        optimizer.step() # Update network parameters

    # Log
    if log:
        writer.add_image('X Img', normalize_tensor(input_img.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('X Seg', normalize_tensor(input_seg.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('X Res', normalize_tensor(img_ano.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('X Ano_grad_act', normalize_tensor(img_ano_act.unsqueeze(1)[:16]), dataformats='NCHW')
        #writer.add_image('X_i out', normalize_tensor(out.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('X_i grad', normalize_tensor(grad.unsqueeze(1)[:16]), dataformats='NCHW')
        #writer.add_histogram('hist-out-torch', out.flatten())
        #writer.add_histogram('hist-grad-torch', grad.flatten())
        #writer.add_histogram('hist-out-mask-torch', out[input_seg > 0].flatten())
        #writer.add_histogram('hist-grad-mask-torch', grad[input_seg > 0].flatten())

        writer.flush()

    del grad
    #del out
    del input_img
    del input_seg
    del img_ano_act
    del img_ano_update

    return img_ano, tot_loss/riter
