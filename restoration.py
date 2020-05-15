__author__ = 'jonatank'

import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import total_variation
from utils.ssim import ssim
from utils.utils import normalize_tensor
from utils.utils import dice_loss, diceloss
from torch.nn import functional as F
import higher

def update_teacher_variables(model, teacher_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.mul_(alpha).add_(1 - alpha, param.data)

def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

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

def run_map_NN(input_img, mask, dec_mu, net, vae_model, riter, device, writer=None, step_size=0.003, log=True):
    # Init params
    input_img = input_img.to(device)
    mask = mask.to(device)
    dec_mu = dec_mu.to(device).float()
    img_ano = nn.Parameter(input_img.clone().to(device), requires_grad=True)

    net.eval()
    for i in range(riter):
        img_ano.detach_()
        img_ano.requires_grad = True

        # Gradient function
        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())
        l2_loss = torch.sum((dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2))

        gfunc = l2_loss + kl_loss

        grad, = torch.autograd.grad(gfunc, img_ano,
                                    grad_outputs=gfunc.data.new(gfunc.shape).fill_(1),
                                    create_graph=True)

        NN_input = torch.stack([input_img, img_ano.detach()]).permute((1, 0, 2, 3)).float()
        out = net(NN_input).squeeze(1)
        img_grad = grad.detach() * out

        img_ano_update = torch.zeros(img_ano.detach().shape).to(device).double()
        img_ano_update[mask > 0] = img_ano.detach()[mask > 0] - step_size * img_grad[mask > 0]
        img_ano.data = img_ano_update

    # Log
    if log:
        writer.add_image('X Img', normalize_tensor(input_img.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('X Res', normalize_tensor(img_ano.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('X_i out', normalize_tensor(out.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('X_i grad', normalize_tensor(grad.unsqueeze(1)[:16]), dataformats='NCHW')
        #if torch.sum(out.flatten()) > 0 and torch.sum(grad.flatten()) > 0:
        #    writer.add_histogram('hist-out-torch', out.flatten())
        #    writer.add_histogram('hist-grad-torch', grad.flatten())

        writer.flush()

    return img_ano

def train_run_map_NN(input_img, dec_mu, net, vae_model, riter, device, writer, input_seg, mask,
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
        #img_ano.requires_grad = True

        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())
        l2_loss = torch.sum((dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2))

        gfunc = l2_loss + kl_loss

        grad, = torch.autograd.grad(gfunc, img_ano, grad_outputs=gfunc.data.new(gfunc.shape).fill_(1),
                                    create_graph=True)

        NN_input = torch.stack([input_img, img_ano]).permute((1, 0, 2, 3)).float()
        out = net(NN_input).squeeze(1)
        img_grad = grad * out

        img_ano_update = torch.zeros(img_ano.detach().shape).to(device).double()

        img_ano_update[mask > 0] = img_ano[mask > 0] - step_size * img_grad[mask > 0]

        img_ano = img_ano_update.clone()

        img_ano_act = torch.tanh(2000*(img_ano_update - input_img).pow(2))

        loss = dice(img_ano_act, input_seg)
        #loss = nn.BCELoss(ano_grad_act.double(), input_seg.double())
        #loss = 1 - ssim(ano_grad_act.unsqueeze(1).float(), input_seg.unsqueeze(1).float())
        loss.backward(retain_graph=True)

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
        writer.add_image('X_i out', normalize_tensor(out.unsqueeze(1)[:16]), dataformats='NCHW')
        writer.add_image('X_i grad', normalize_tensor(grad.unsqueeze(1)[:16]), dataformats='NCHW')
        #writer.add_histogram('hist-out-torch', out.flatten())
        #writer.add_histogram('hist-grad-torch', grad.flatten())
        #writer.add_histogram('hist-out-mask-torch', out[input_seg > 0].flatten())
        #writer.add_histogram('hist-grad-mask-torch', grad[input_seg > 0].flatten())

        writer.flush()

    del grad
    del out
    del input_img
    del input_seg
    del img_ano_act
    del img_ano_update

    return img_ano, tot_loss/riter

def train_run_map_NN_teacher(input_img, input_img_teacher, dec_mu, dec_mu_teacher, net, net_teacher, vae_model, riter,
                             device, writer, input_seg, input_seg_teacher, epoch, optimizer=None, step_size=0.003,
                             train=True, teacher_decay=0.999, consistency_weight=1):
    # Init params
    input_img = nn.Parameter(input_img, requires_grad=False)
    dec_mu = nn.Parameter(dec_mu.to(device).float(), requires_grad=False)
    img_ano = nn.Parameter(input_img.clone().to(device),requires_grad=True)
    input_seg = input_seg.to(device)

    # Init MAP Optimizer
    MAP_optimizer = optim.Adam([img_ano], lr=step_size)

    if train:
        net.train()
        net_teacher.train()
        optimizer.zero_grad()

    dice = diceloss()
    #BCE = nn.BCELoss()

    tot_dice_loss = 0
    tot_consistency_loss = 0
    tot_loss = 0

    for i in range(riter):
        MAP_optimizer.zero_grad()
        # Gradient step MAP
        __, z_mean, z_cov, __ = vae_model(img_ano.unsqueeze(1).double())

        # Define G function
        l2_loss = (dec_mu.view(-1, dec_mu.numel()) - img_ano.view(-1, img_ano.numel())).pow(2)
        kl_loss = -0.5 * torch.sum(1 + z_cov - z_mean.pow(2) - z_cov.exp())

        gfunc = torch.sum(l2_loss) + kl_loss
        gfunc.backward()

        NN_input = torch.stack([input_img, img_ano]).permute((1, 0, 2, 3)).float()
        out = net(NN_input)
        out_teacher = net_teacher(NN_input)

        out_teacher = nn.Parameter(out_teacher.detach().data, requires_grad=False)
        consistency_loss = consistency_weight * symmetric_mse_loss(out, out_teacher)

        img_ano.grad += out.squeeze(1)
        ano_grad_act = 1 - 2 * torch.sigmoid(-10000 * img_ano.grad.pow(2))
        dice_loss = dice(ano_grad_act, input_seg)

        loss = dice_loss + consistency_loss
        # loss = BCE(ano_grad_act.double(), input_seg.double())
        # loss = 1 - ssim(ano_grad_act.unsqueeze(1).float(), input_seg.unsqueeze(1).float())

        loss.backward()
        MAP_optimizer.step() # Gradient step to update img_ano

        # Log
        tot_loss += loss.item()
        tot_dice_loss += dice_loss.item()
        tot_consistency_loss += consistency_loss.item()


    if train:
        optimizer.step() # Update network parameters
        update_teacher_variables(net, net_teacher, teacher_decay, epoch) # Update teacher network parameters

    # Log
    writer.add_scalar('Dice Loss', tot_dice_loss/riter)
    writer.add_scalar('Consistency Loss', tot_consistency_loss/riter)
    writer.add_image('X Img', normalize_tensor(input_img.unsqueeze(1)[:16]), dataformats='NCHW')
    writer.add_image('X Seg', normalize_tensor(input_seg.unsqueeze(1)[:16]), dataformats='NCHW')
    writer.add_image('X Ano_grad_act', normalize_tensor(ano_grad_act.unsqueeze(1)[:16]), dataformats='NCHW')
    writer.add_image('X_i ano_img', normalize_tensor(img_ano.unsqueeze(1)[:16]), dataformats='NCHW')
    writer.add_image('X_i out', normalize_tensor(out[:16]), dataformats='NCHW')
    writer.add_image('X_i ano_img_grad', normalize_tensor(img_ano.grad.unsqueeze(1)[:16]), dataformats='NCHW')

    del input_img
    del input_seg
    del out
    del out_teacher
    del ano_grad_act
    del img_ano.grad

    return img_ano, tot_loss/riter