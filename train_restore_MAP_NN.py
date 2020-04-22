import numpy as np
from skimage.transform import resize

import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from restoration import train_run_map_NN
from models.shallow_UNET import shallow_UNet
from models.unet import UNet
from models.covnet import ConvNet
from datasets import brats_dataset_subj, brats_dataset
from utils.auc_score import compute_tpr_fpr
from utils import threshold
import pickle
import argparse
import yaml
import random
from utils.utils import normalize_tensor
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=0)
    parser.add_argument("--config", required=True, help="Path to config")
    parser.add_argument('--subjs', type=int, required=True, help="Number of subjects")

    opt = parser.parse_args()
    name = opt.name
    subj_nbr = opt.subjs

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    model_name = config['vae_name']
    data_path = config['path']
    riter = config['riter']
    batch_size = config["batch_size"]
    img_size = config["spatial_size"]
    lr_rate = float(config['lr_rate'])
    step_rate = float(config['step_rate'])
    log_freq = config['log_freq']
    original_size = config['orig_size']
    log_dir = config['log_dir']
    n_latent_samples = 25
    epochs = config['epochs']

    print('Name: ', name, 'Lr_rate: ', lr_rate, ' Riter: ', riter, ' Subjs: ', subj_nbr)

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    # Load trained vae model
    vae_path = '/scratch_net/biwidl214/jonatank/logs/vae/'
    path = vae_path + model_name + '.pth'
    vae_model = torch.load(path, map_location=torch.device(device))
    vae_model.eval()

    # Create guiding net
    net = shallow_UNet(name, 2, 1, 16).to(device)
    #net = ConvNet(name, 2, 1, 8).to(device)
    #net = UNet(name, 2, 1, 4).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)

    # Load list of subjects
    f = open(data_path + 'subj_t2_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list_all = list(subj_dict.keys())
    random.shuffle(subj_list_all)
    subj_list = subj_list_all[:subj_nbr-1]
    print(subj_list)

    # Init logging with Tensorboard
    writer = SummaryWriter(log_dir + name)

    validation = True

    subj_val_list = []
    subj_val_list.append(subj_list_all[subj_nbr])
    print('validation subject', subj_val_list)
    writer_valid = SummaryWriter(log_dir + 'valid_' + name)

    for ep in range(epochs):
        # Metrics init
        TP = 0
        FN = 0
        FP = 0
        y_pred = []
        y_true = []
        subj_dice = []

        for subj in subj_list: # Iterate every subject
            slices = subj_dict[subj] # Slices for each subject CHANGE

            # Load data
            subj_dataset = brats_dataset_subj(data_path, 'train', img_size, slices, use_aug=True)
            subj_loader = data.DataLoader(subj_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
            print('Subject ', subj, ' Number of Slices: ', subj_dataset.size)

            for batch_idx, (scan, seg, mask) in enumerate(subj_loader):
                scan = scan.double().to(device)
                decoded_mu = torch.zeros(scan.size())

                # Get average prior
                for s in range(n_latent_samples):
                    with torch.no_grad():
                        recon_batch, z_mean, z_cov, res = vae_model(scan)
                    decoded_mu += np.array([1 * recon_batch[i].detach().cpu().numpy() for i in range(scan.size()[0])])

                decoded_mu = decoded_mu / n_latent_samples

                # Remove channel
                scan = scan.squeeze(1)
                seg = seg.squeeze(1)
                mask = mask.squeeze(1).cpu().detach().numpy()

                #train_riter = np.random.randint(1, 100)
                restored_batch = train_run_map_NN(scan, decoded_mu, net, vae_model, riter, device, writer, seg,
                                                  optimizer, step_rate, log_freq)

                seg = seg.cpu().detach().numpy()
                # Predicted abnormalty is difference between restored and original batch
                error_batch = np.zeros([scan.size()[0],original_size,original_size])
                restored_batch_resized = np.zeros([scan.size()[0],original_size,original_size])

                for idx in range(scan.size()[0]): # Iterate trough for resize
                    error_batch[idx] = resize(abs(scan[idx] - restored_batch[idx]).cpu().detach().numpy(), (200,200))
                    restored_batch_resized[idx] = resize(restored_batch[idx].cpu().detach().numpy(), (200,200))

                # Remove preds and seg outside mask and flatten
                mask = resize(mask, (scan.size()[0], original_size, original_size))
                seg = resize(seg, (scan.size()[0], original_size, original_size))

                error_batch_m = error_batch[mask > 0].ravel()
                seg_m = seg[mask > 0].ravel().astype(bool)

                # AUC
                y_pred.extend(error_batch_m.tolist())
                y_true.extend(seg_m.tolist())

                # DICE
                '''
                # Create binary prediction map
                error_batch_m[error_batch_m >= thr_error] = 1
                error_batch_m[error_batch_m < thr_error] = 0

                # Calculate and sum total TP, FN, FP
                TP += np.sum(seg_m[error_batch_m == 1])
                FN += np.sum(seg_m[error_batch_m == 0])
                FP += np.sum(error_batch_m[seg_m == 0])
                '''

            AUC = roc_auc_score(y_true, y_pred)
            print('AUC : ', AUC)
            writer.add_scalar('AUC:', AUC)

            '''
            dice = (2 * TP) / (2 * TP + FN + FP)
            subj_dice.append(dice)
            print('DCS: ', dice)
            writer.add_scalar('Dice:', dice)
            '''
            writer.flush()

        if ep % log_freq == 0:
            # Save model
            path = '/scratch_net/biwidl214/jonatank/logs/restore/' + name + str(ep) + '.pth'
            torch.save(net, path)

            ## Write to tensorboard
            writer.add_image('Batch of Scan', scan.unsqueeze(1)[:16], batch_idx, dataformats='NCHW')
            writer.add_image('Batch of Restored', normalize_tensor(np.expand_dims(restored_batch_resized, axis=1)[:16]),
                             batch_idx, dataformats='NCHW')
            writer.add_image('Batch of Diff Restored Scan', normalize_tensor(np.expand_dims(error_batch, axis=1)[:16]),
                             batch_idx, dataformats='NCHW')
            writer.add_image('Batch of Ground truth', np.expand_dims(seg, axis=1)[:16], batch_idx, dataformats='NCHW')
            writer.flush()

        ## VALIDATION
        if validation:
            y_pred_valid = []
            y_true_valid = []

            for subj in subj_val_list:  # Iterate every subject
                slices = subj_dict[subj]  # Slices for each subject CHANGE

                # Load data
                subj_dataset = brats_dataset_subj(data_path, 'train', img_size, slices, use_aug=True)
                subj_loader = data.DataLoader(subj_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
                print('Subject ', subj, ' Number of Slices: ', subj_dataset.size)

                loss = 0
                for batch_idx, (scan, seg, mask) in enumerate(subj_loader):
                    scan = scan.double().to(device)
                    decoded_mu = torch.zeros(scan.size())

                    # Get average prior
                    for s in range(n_latent_samples):
                        with torch.no_grad():
                            recon_batch, z_mean, z_cov, res = vae_model(scan)
                        decoded_mu += np.array(
                            [1 * recon_batch[i].detach().cpu().numpy() for i in range(scan.size()[0])])

                    decoded_mu = decoded_mu / n_latent_samples

                    # Remove channel
                    scan = scan.squeeze(1)
                    seg = seg.squeeze(1)
                    mask = mask.squeeze(1).cpu().detach().numpy()

                    # train_riter = np.random.randint(1, 100)
                    restored_batch = train_run_map_NN(scan, decoded_mu, net, vae_model, riter, device, writer_valid,
                                                      seg, step_size=step_rate, log_freq=log_freq, train=False)

                    seg = seg.cpu().detach().numpy()
                    # Predicted abnormalty is difference between restored and original batch
                    error_batch = np.zeros([scan.size()[0], original_size, original_size])

                    for idx in range(scan.size()[0]):  # Iterate trough for resize
                        error_batch[idx] = resize(abs(scan[idx] - restored_batch[idx]).cpu().detach().numpy(),
                                                  (200, 200))

                    # Remove preds and seg outside mask and flatten
                    mask = resize(mask, (scan.size()[0], original_size, original_size))
                    seg = resize(seg, (scan.size()[0], original_size, original_size))

                    error_batch_m = error_batch[mask > 0].ravel()
                    seg_m = seg[mask > 0].ravel().astype(bool)

                    # AUC
                    y_pred_valid.extend(error_batch_m.tolist())
                    y_true_valid.extend(seg_m.tolist())

            AUC = roc_auc_score(y_true_valid, y_pred_valid)
            print('AUC Valid: ', AUC)
            writer_valid.add_scalar('AUC:', AUC)
            writer.flush()

