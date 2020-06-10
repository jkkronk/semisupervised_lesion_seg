__author__ = 'jonatank'
import sys
sys.path.insert(0, '/scratch_net/biwidl214/jonatank/code_home/restor_MAP/')

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.unet import UNET, train_unet, valid_unet
import argparse
import yaml
import pickle
import numpy as np
import random

from datasets import brats_dataset_subj, camcan_dataset

if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=0)
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--subjs", type=float, help="Number of subjects for training")
    parser.add_argument("--aug", type=int)

    opt = parser.parse_args()
    model_name = opt.model_name
    subj_nbr = opt.subjs
    aug = bool(opt.aug)

    with open(opt.config) as f:
        config = yaml.safe_load(f)

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

    # Load healthy subjects
    print("Loading healthy data...")
    path_2_healthy = '/scratch_net/biwidl214/jonatank/data/dataset_abnormal/new/camcan/'
    healthy_train_dataset = camcan_dataset(path_2_healthy, True, img_size, data_aug=1)
    healthy_train_data_loader = data.DataLoader(healthy_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Healthy train data loaded')

    # Load list of subjects
    f = open(data_path + 'subj_t2_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list_all = list(subj_dict.keys())
    random.shuffle(subj_list_all)
    subj_list = subj_list_all[:subj_nbr]
    #if subj_nbr == 1:
    #    subj_list = ['Brats17_CBICA_BFB_1_t2_unbiased.nii.gz']

    print(subj_list)

    slices = []
    for subj in subj_list:  # Iterate every subject
        slices.extend(subj_dict[subj])  # Slices for each subject

    # Load data
    subj_dataset = brats_dataset_subj(data_path, 'train', img_size, slices, use_aug=True)
    subj_loader = data.DataLoader(subj_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    print('Subject ', subj, ' Number of Slices: ', subj_dataset.size)

    # Load list of validation subjects
    f = open(data_path + 'subj_t2_valid_dict.pkl', 'rb')
    val_subj_dict = pickle.load(f)
    f.close()

    val_subj_list_all = list(val_subj_dict.keys())
    # if subj_nbr == 1:
    #    subj_list = ['Brats17_CBICA_BFB_1_t2_unbiased.nii.gz']

    print(val_subj_list_all)

    slices = []
    for subj in val_subj_list_all:  # Iterate every subject
        slices.extend(val_subj_dict[subj])  # Slices for each subject

    # Load validation data
    val_subj_dataset = brats_dataset_subj(data_path, 'train', img_size, slices, use_aug=False)
    val_subj_loader = data.DataLoader(subj_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    print('Subject ', subj, ' Number of Slices: ', subj_dataset.size)

    # Create unet
    model = UNET(model_name, 1,1,32).to(device)

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # Tensorboard Init
    writer_train = SummaryWriter(log_dir + model.name + '_train')
    writer_valid = SummaryWriter(log_dir + model.name + '_valid')

    # Start training
    print('Start training:')
    for epoch in range(epochs):
        print('Epoch:', epoch)

        loss = train_unet(model, subj_loader, device, optimizer)
        loss_valid = valid_unet(model, val_subj_loader, device)

        writer_train.add_scalar('Loss',loss, epoch)
        writer_train.flush()
        writer_valid.add_scalar('Loss',loss_valid, epoch)
        writer_valid.flush()

        if epoch % log_freq == 0 and not epoch == 0:
            data_path = log_model + model_name + str(epoch) + '.pth'
            torch.save(model, data_path)