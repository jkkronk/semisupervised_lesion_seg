import sys
sys.path.insert(0, '/scratch_net/biwidl214/jonatank/code_home/restor_MAP/')

import os
import yaml
import numpy as np
from skimage.transform import resize
from sklearn.metrics import roc_auc_score

import pickle
import argparse
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from baselines.restore_TVnorm.resotration import run_map_TV
from datasets import brats_dataset_subj
from utils import threshold

if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=0)
    parser.add_argument("--config", required=True, help="Path to config")
    parser.add_argument("--fprate", type=float, help="False positive rate")
    parser.add_argument('--vae', type=str, default=0)

    opt = parser.parse_args()
    name = opt.name
    fprate = opt.fprate
    model_name = opt.vae

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    weight = config['weight']
    #model_name = config['vae_name'] # camcan_400_Aug_2_100
    data_path = config['path']
    riter = config['riter']
    batch_size = config["batch_size"]
    img_size = config["spatial_size"]
    step_rate = config['step_rate']
    log_freq = config['log_freq']
    original_size = config['orig_size']
    log_dir = config['log_dir']
    n_latent_samples = 25
    preset_threshold = [] #[0.0907, 0.0381, 0.0810]

    print(' Vae model: ', model_name, ' Fprate: ',fprate)

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    vae_path = '/scratch_net/biwidl214/jonatank/logs/vae/'
    # Load trained vae model
    path = vae_path + model_name + '.pth'
    vae_model = torch.load(path, map_location=torch.device(device))
    vae_model.eval()

    # Compute threshold with help of camcan set
    if not preset_threshold:
        thr_error = \
            threshold.compute_threshold_TV(fprate, vae_model, img_size, batch_size, n_latent_samples,
                              device, riter, step_rate, weight, renormalized=True, n_random_sub=25)
    else:
        thr_error = preset_threshold
    print(thr_error)

    # Load list of subjects
    f = open(data_path + 'subj_t2_test_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list = list(subj_dict.keys())

    # Init logging with Tensorboard
    writer = SummaryWriter(log_dir + name)

    subj_dice = []
    thresh_error = []
    total_p = 0
    total_n = 0
    auc_error_tot = 0

    y_pred = []
    y_true = []

    for subj in subj_list: # Iterate every subject
        # Metrics init
        TP = 0
        FN = 0
        FP = 0

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
            mask = mask.squeeze(1).cpu().detach().numpy()
            seg = seg.squeeze(1).cpu().detach().numpy()

            restored_batch = run_map_TV(scan, decoded_mu, vae_model, riter, device, weight, step_rate)

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

            # for AUC
            y_pred = np.append(y_pred, error_batch_m)
            y_true = np.append(y_true, seg_m)

            # DICE
            error_batch_m_b = np.copy(error_batch_m)
            # Create binary prediction map
            error_batch_m_b[error_batch_m >= thr_error] = 1
            error_batch_m_b[error_batch_m < thr_error] = 0

            # Calculate and sum total TP, FN, FP
            TP += np.sum(seg_m[error_batch_m_b == 1])
            FN += np.sum(seg_m[error_batch_m_b == 0])
            FP += np.sum(error_batch_m_b[seg_m == 0])

        # AUC
        if not all(element == 0 for element in y_true):
            AUC = roc_auc_score(y_true, y_pred)
        print('AUC : ', AUC)
        writer.add_scalar('AUC:', AUC)
        writer.flush()

        dice =  (2*TP)/(2*TP+FN+FP)

        print('DCS: ', dice)
        writer.add_scalar('Dice:', dice)
        writer.flush()
        subj_dice.append(dice)

    if not all(element == 0 for element in y_true):
        AUC = roc_auc_score(y_true, y_pred)
    print('to AUC: ', AUC)
    writer.add_scalar('AUC:', AUC)
    writer.flush()

    print('Dice mean: ', np.mean(np.array(subj_dice), axis=0), ' std: ', np.std(np.array(subj_dice), axis=0))
