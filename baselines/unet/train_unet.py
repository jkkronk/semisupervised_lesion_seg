__author__ = 'jonatank'
import torch
from torchvision import utils
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from unet import UNet, train_unet, valid_unet
import argparse
import yaml
import numpy as np

from datasets import brats_dataset

if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=0)
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--prop_subj", type=float, help="Procentage subjects of full training set")
    parser.add_argument("--aug", type=int)

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    model_name = opt.model_name
    prop_train_subjects = opt.prop_subj
    aug = bool(opt.aug)

    lr_rate = float(config['lr_rate'])
    data_path = config['path']
    epochs = config['epochs']
    batch_size = config["batch_size"]
    img_size = config["spatial_size"]
    log_freq = config['log_freq']
    log_dir = config['log_dir']
    log_model = config['model_dir']

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    # Load data
    train_dataset = brats_dataset(data_path, 'train', img_size, prop_subjects=prop_train_subjects, use_aug=aug)
    train_data_loader  = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=False)
    print('TRAIN loaded: Slices: ' + str(len(train_data_loader.dataset)) + ' Subjects: ' + str(int(200*prop_train_subjects)))
    print(train_dataset.keys)

    valid_dataset = brats_dataset(data_path, 'valid', img_size, prop_subjects=1)
    valid_data_loader  = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=False)
    print('VALIDATION loaded: Slices: ' + str(len(valid_data_loader.dataset)) + ' Subjects: ' + str(int(35*1)))

    # Create unet
    model = UNet(model_name, 1,1,32).to(device)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr_rate//100)

    # Tensorboard Init
    writer_train = SummaryWriter(log_dir + model.name + '_train')
    writer_valid = SummaryWriter(log_dir + model.name + '_valid')

    # Start training
    print('Start training:')
    for epoch in range(epochs):
        print('Epoch:', epoch)

        loss = train_unet(model, train_data_loader, device, optimizer)
        loss_valid = valid_unet(model, valid_data_loader, device)

        # Cosine annealing
        scheduler.step()

        writer_train.add_scalar('Loss',loss, epoch)
        writer_train.flush()
        writer_valid.add_scalar('Loss',loss_valid, epoch)
        writer_valid.flush()

        if epoch % log_freq == 0 and not epoch == 0:
            data_path = log_model + model_name + str(epoch) + '.pth'
            torch.save(model, data_path)