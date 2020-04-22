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
import random
from utils.utils import normalize_tensor
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=0)
    parser.add_argument("--config", required=True, help="Path to config")
    parser.add_argument("--fprate", type=float, help="False positive rate")
    parser.add_argument("--netname", type=str, help="Net name of guiding net")

    opt = parser.parse_args()
    name = opt.name
    fprate = opt.fprate
    net_name = opt.netname

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    model_name = config['vae_name']
    #net_name = config['net_name']
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
    preset_threshold = [] #1.6875
    epochs = config['epochs']

    print(' Vae model: ', model_name, ' NN model: ', net_name)

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    # Load trained vae model
    vae_path = '/scratch_net/biwidl214/jonatank/logs/vae/'
    path = vae_path + model_name + '.pth'
    vae_model = torch.load(path, map_location=torch.device(device))
    vae_model.eval()

    # Load trained nn model
    path = log_dir + net_name + '.pth'
    net = torch.load(path, map_location=torch.device(device))
    net.eval()

    # Compute threshold with help of camcan set
    if not preset_threshold:
        thr_error = \
            threshold.compute_threshold_subj(data_path, vae_model, net, img_size,
                                             ['Brats17_2013_11_1_t2_unbiased.nii.gz'], batch_size, n_latent_samples,
                                             device, riter, step_rate)
    else:
        thr_error = preset_threshold
    print(thr_error)

    # Load list of subjects
    f = open(data_path + 'subj_t2_test_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list = list(subj_dict.keys())
    #random.shuffle(subj_list)
    #subj_list = subj_list[]

    # Init logging with Tensorboard
    writer = SummaryWriter(log_dir + name)

    # Metrics init
    y_true = []
    y_pred = []
    subj_dice = []

    for i, subj in enumerate(subj_list):  # Iterate every subject
        TP = 0
        FN = 0
        FP = 0

        print(i/len(subj_list))
        slices = subj_dict[subj]  # Slices for each subject

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
            scan = scan.squeeze(1)
            seg = seg.squeeze(1)
            mask = mask.squeeze(1)

            restored_batch = run_map_NN(scan, decoded_mu, net, vae_model, riter, device, writer, step_size=step_rate, input_seg=seg)

            seg = seg.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()

            # Predicted abnormalty is difference between restored and original batch
            error_batch = np.zeros([scan.size()[0], original_size, original_size])
            restored_batch_resized = np.zeros([scan.size()[0], original_size, original_size])

            for idx in range(scan.size()[0]):  # Iterate trough for resize
                error_batch[idx] = resize(abs(scan[idx] - restored_batch[idx]).cpu().detach().numpy(), (200, 200))
                restored_batch_resized[idx] = resize(restored_batch[idx].cpu().detach().numpy(), (200, 200))

            # Remove preds and seg outside mask and flatten
            mask = resize(mask, (scan.size()[0], original_size, original_size))
            seg = resize(seg, (scan.size()[0], original_size, original_size))

            # AUC
            error_batch_m = error_batch[mask > 0].ravel()
            y_pred.extend(error_batch_m.tolist())

            seg_m = seg[mask > 0].ravel().astype(int)
            y_true.extend(seg_m.tolist())

            print(error_batch_m.max(), error_batch_m.min())

            # DICE
            # Create binary prediction map
            error_batch_m[error_batch_m >= thr_error] = 1
            error_batch_m[error_batch_m < thr_error] = 0

            # Calculate and sum total TP, FN, FP
            TP += np.sum(seg_m[error_batch_m == 1])
            FN += np.sum(seg_m[error_batch_m == 0])
            FP += np.sum(error_batch_m[seg_m == 0])

        AUC = roc_auc_score(y_true, y_pred)
        print('AUC : ', AUC)
        writer.add_scalar('AUC:', AUC)

        dice = (2*TP)/(2*TP+FN+FP)
        subj_dice.append(dice)
        print('DCS: ', dice)
        writer.add_scalar('Dice:', dice)
        writer.flush()

        ## Write to tensorboard
        writer.add_image('Batch of Scan', scan.unsqueeze(1)[:16], batch_idx, dataformats='NCHW')
        writer.add_image('Batch of Restored', normalize_tensor(np.expand_dims(restored_batch_resized, axis=1)[:16]),
                         batch_idx, dataformats='NCHW')
        writer.add_image('Batch of Diff Restored Scan', normalize_tensor(np.expand_dims(error_batch, axis=1)[:16]),
                         batch_idx, dataformats='NCHW')
        writer.add_image('Batch of Ground truth', np.expand_dims(seg, axis=1)[:16], batch_idx, dataformats='NCHW')

        writer.flush()

    avrg_dcs = sum(subj_dice) / len(subj_dice)
    print('DCS: ',  avrg_dcs)
    writer.add_scalar('Dice:', avrg_dcs)
    writer.flush()
