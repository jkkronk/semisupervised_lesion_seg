import numpy as np
from skimage.transform import resize

import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from restoration import run_map_NN
from models.shallow_UNET import shallow_UNet
from datasets import brats_dataset_subj
from utils.auc_score import compute_tpr_fpr
from utils import threshold
import pickle
import argparse
import yaml

if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=0)
    parser.add_argument("--config", required=True, help="Path to config")
    parser.add_argument("--fprate", type=float, help="False positive rate")

    opt = parser.parse_args()
    name = opt.name
    fprate = opt.fprate

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    model_name = config['vae_name']
    data_path = config['path']
    riter = config['riter']
    batch_size = config["batch_size"]
    img_size = config["spatial_size"]
    lr_rate = float(config['lr_rate'])
    step_rate = config['step_rate']
    log_freq = config['log_freq']
    original_size = config['orig_size']
    log_dir = config['log_dir']
    n_latent_samples = 25
    preset_threshold = [0.0907, 0.0381, 0.0810]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    # Load trained vae model
    path = 'models/' + model_name + '.pth'
    vae_model = torch.load(path, map_location=torch.device(device))
    vae_model.eval()

    # Create shallow UNET
    net = shallow_UNet(name, 2, 1, 32).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)

    # Compute threshold with help of camcan set
    if not preset_threshold:
        thr_error, thr_error_corr, thr_MAD = \
            threshold.compute_threshold(fprate, vae_model, img_size, batch_size, n_latent_samples,
                              device, renormalized=True, n_random_sub=100)
    else:
        thr_error, thr_error_corr, thr_MAD = preset_threshold
    print(thr_error, thr_error_corr, thr_MAD)

    # Load list of subjects
    f = open(data_path + 'subj_t2_test_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list = list(subj_dict.keys())

    # Init logging with Tensorboard
    writer = SummaryWriter(log_dir + name)

    subj_dice = []
    subj_AUC = []


    for subj in subj_list: # Iterate every subject
        # Metrics init
        TP = 0
        FN = 0
        FP = 0
        thresh_error = []
        total_p = 0
        total_n = 0
        auc_error_tot = 0

        slices = subj_dict[subj] # Slices for each subject

        # Load data
        subj_dataset = brats_dataset_subj(data_path, 'test', img_size, slices)  # Change rand_subj to True
        subj_loader = data.DataLoader(subj_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        print('Subject ', subj, ' Number of Slices: ', subj_dataset.size)

        for batch_idx, (scan, seg, mask) in enumerate(subj_loader):
            scan = scan.double().to(device)
            decoded_mu = torch.zeros(scan.size())

            # Get average prior
            for s in range(n_latent_samples):
                recon_batch, z_mean, z_cov, res = vae_model(scan)
                decoded_mu += np.array([1 * recon_batch[i].detach().cpu().numpy() for i in range(scan.size()[0])])

            decoded_mu = decoded_mu / n_latent_samples

            # Remove channel
            decoded_mu = decoded_mu.squeeze(1)
            scan = scan.squeeze(1)
            seg = seg.squeeze(1)
            mask = mask.squeeze(1).cpu().detach().numpy()

            if np.sum(mask)>30000: #30000 # Only restore when there is enough brain tissue
                #print('restoring')
                restored_batch = run_map_NN(scan, decoded_mu, net, riter, device, writer, optimizer, seg, 'train', step_rate)
            else: # Else scan is only background
                restored_batch = scan
                pred_res_restored = torch.zeros((img_size,img_size))

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
            seg_m = seg[mask > 0].ravel()

            # AUC
            if not len(thresh_error): # Create total error list
                thresh_error = np.concatenate((np.sort(error_batch_m[::100]), [15]))
                error_tprfpr = np.zeros((2, len(thresh_error)))

            # Compute true positive rate and false positve rate
            error_tprfpr += compute_tpr_fpr(seg_m, error_batch_m, thresh_error)

            # Number of total positive and negative in segmentation
            total_p += np.sum(seg_m > 0)
            total_n += np.sum(seg_m == 0)

            # TP-rate and FP-rate calculation
            tpr_error = error_tprfpr[0] / total_p
            fpr_error = error_tprfpr[1] / total_n

            # Add to total AUC
            auc_error = 1. + np.trapz(fpr_error, tpr_error)

            # DICE
            # Create binary prediction map
            error_batch_m[error_batch_m >= thr_error_corr] = 1
            error_batch_m[error_batch_m < thr_error_corr] = 0

            # Calculate and sum total TP, FN, FP
            TP += np.sum(seg_m[error_batch_m == 1])
            FN += np.sum(seg_m[error_batch_m == 0])
            FP += np.sum(error_batch_m[seg_m == 0])

        print('AUC: ', auc_error)
        writer.add_scalar('AUC:', auc_error, subj)
        writer.flush()
        subj_AUC.append(auc_error)

        dice =  (2*TP)/(2*TP+FN+FP)

        print('DCS: ', dice)
        writer.add_scalar('Dice:', dice, subj)
        writer.flush()
        subj_dice.append(dice)

    print('Dice mean: ', np.mean(np.array(subj_dice), axis=0), ' std: ', np.std(np.array(subj_dice), axis=0))
    print('AUC mean: ', np.mean(np.array(subj_AUC), axis=0), ' std: ', np.std(np.array(subj_AUC), axis=0))

######
#if batch_idx % log_freq == 0:
#    # Write to tensorboard
#    writer.add_image('Batch of Scan', scan.unsqueeze(1)[:16], batch_idx, dataformats='NCHW')
#   writer.add_image('Batch of Restored', np.clip(np.expand_dims(restored_batch_resized, axis=1), 0, 1)[:16], batch_idx,
#                    dataformats='NCHW')
#    writer.add_image('Batch of Diff Restored Scan', normalize_tensor(np.expand_dims(error_batch, axis=1)[:16]),
#                     batch_idx, dataformats='NCHW')
#    writer.add_image('Batch of Ground truth', np.expand_dims(seg, axis=1)[:16], batch_idx, dataformats='NCHW')
#    writer.flush()