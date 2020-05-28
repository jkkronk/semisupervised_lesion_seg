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
from sklearn import metrics
import yaml
import random
from utils.utils import normalize_tensor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


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
    batch_size = 32 #config["batch_size"]
    img_size = config["spatial_size"]
    lr_rate = float(config['lr_rate'])
    step_rate = float(config['step_rate'])
    log_freq = config['log_freq']
    original_size = config['orig_size']
    log_dir = config['log_dir']
    n_latent_samples = 25
    preset_threshold = []
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
    #
    #Brats17_TCIA_462_1_t2_unbiased.nii.gz
    train_subjs = ['Brats17_TCIA_231_1_t2_unbiased.nii.gz']

    # Compute threshold with help of camcan set
    if not preset_threshold:
        if fprate == 0:
            thr_error = \
                threshold.compute_threshold_subj(data_path, vae_model, net, img_size,
                                             train_subjs, batch_size, n_latent_samples,
                                             device, riter, step_rate)
        else:
            print('Healthy!')
            thr_error = threshold.compute_threshold(fprate, vae_model, img_size, batch_size, n_latent_samples, device, n_random_sub=5,
                          net_model=net, riter=500, step_size=step_rate,
                          renormalized=False)
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
    tot_error_m = np.array([])
    tot_seg_m = np.array([])
    subj_dice = np.array([])
    total_p = 0
    total_n = 0
    thresh_error = []
    tot_AUC = []
    y_pred = []
    y_true = []

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

            restored_batch = run_map_NN(scan, mask, decoded_mu, net, vae_model, riter, device, seg, thr_error, writer,
                                        step_size=step_rate, log=bool(batch_idx % 3))

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

            error_batch_m = error_batch[mask > 0].ravel()
            seg_m = seg[mask > 0].ravel().astype(bool)

            tot_error_m = np.append(tot_error_m,error_batch_m)
            tot_seg_m = np.append(tot_seg_m,seg_m)

            y_pred = np.append(y_pred, error_batch_m)
            y_true = np.append(y_true, seg_m)

            # DICE
            # Create binary prediction map
            error_batch_m[error_batch_m >= thr_error] = 1
            error_batch_m[error_batch_m < thr_error] = 0

            # Calculate and sum total TP, FN, FP
            TP += np.sum(seg_m[error_batch_m == 1])
            FN += np.sum(seg_m[error_batch_m == 0])
            FP += np.sum(error_batch_m[seg_m == 0])

            if np.sum(seg_m) == 0:
                seg_m[0] = 1 # Hacky way and not good

        auc_error = roc_auc_score(y_true, y_pred)
        ## evaluate AUC for ROC using universal thresholds
        '''
        if not len(thresh_error):
            thresh_error = np.concatenate((np.sort(tot_error_m[::100]), [15]))
            error_tprfpr = np.zeros((2, len(thresh_error)))

        error_tprfpr += compute_tpr_fpr(tot_seg_m, tot_error_m, thresh_error)

        total_p += np.sum(tot_seg_m == 1)
        total_n += np.sum(tot_seg_m == 0)

        tpr_error = error_tprfpr[0] / total_p
        fpr_error = error_tprfpr[1] / total_n

        auc_error = 1. + np.trapz(fpr_error, tpr_error)
        '''
        print('AUC : ', auc_error)
        writer.add_scalar('AUC:', auc_error)
        tot_AUC = np.append(tot_AUC, auc_error)

        dice = (2*TP)/(2*TP+FN+FP)
        subj_dice = np.append(subj_dice, dice)
        #subj_dice.append(dice)
        print('DCS: ', dice)
        writer.add_scalar('Dice:', dice)
        writer.flush()

        ## Write to tensorboard
        #writer.add_image('Batch of Scan', scan.unsqueeze(1)[:16], batch_idx, dataformats='NCHW')
        #writer.add_image('Batch of Restored', normalize_tensor(np.expand_dims(restored_batch_resized, axis=1)[:16]),
        #                 batch_idx, dataformats='NCHW')
        #writer.add_image('Batch of Diff Restored Scan', normalize_tensor(np.expand_dims(error_batch, axis=1)[:16]),
        #                 batch_idx, dataformats='NCHW')
        #writer.add_image('Batch of Ground truth', np.expand_dims(seg, axis=1)[:16], batch_idx, dataformats='NCHW')
        #writer.flush()


    #AUC = roc_auc_score(y_pred.tolist(), y_true.tolist())
    #print('AUC TEST SET: ', AUC)
    mean_AUC = np.mean(tot_AUC)
    std_AUC = np.std(tot_AUC)
    print('Mean All AUC: ', mean_AUC)
    print('Std ALL AUC: ', std_AUC)
    auc_error = roc_auc_score(y_true, y_pred)
    print('All AUC: ', auc_error)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)

    roc_auc = metrics.auc(fpr, tpr)

    mean_dcs = np.mean(subj_dice)
    std_dcs = np.std(subj_dice)
    print('Mean All DCS: ',  mean_dcs)
    print('Std ALL DCS: ', std_dcs)
    writer.add_scalar('Dice:', mean_dcs)
    writer.flush()

    ix = np.where(thresholds == thr_error)

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(fpr[ix], tpr[ix], 'ro')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('testAUC.png')
