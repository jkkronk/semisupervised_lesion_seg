import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import total_variation

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