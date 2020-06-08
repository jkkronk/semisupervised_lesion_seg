import numpy as np
from skimage.transform import resize

import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from restoration import train_run_map_NN_teacher, train_run_map_NN
from models.shallow_UNET import shallow_UNet
from models.unet import UNet
from models.covnet import ConvNet
from datasets import brats_dataset_subj, brats_dataset_subj_teacher
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
    parser.add_argument('--K_actf', type=int, default=1, help="Activation param")

    opt = parser.parse_args()
    name = opt.name
    subj_nbr = opt.subjs
    K_actf = opt.K_actf

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

    use_teacher = False
    validation = False

    print('Name: ', name, 'Lr_rate: ', lr_rate, 'Use Teacher: ', use_teacher,' Riter: ', riter, ' Subjs: ', subj_nbr)

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
    #net = ConvNet(name, 2, 1, 4).to(device)
    #net = UNet(name, 2, 1, 4).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)

    # Create mean teacher
    if use_teacher:
        net_teacher = shallow_UNet(name, 2, 1, 16).to(device)

    # Load list of subjects
    f = open(data_path + 'subj_t2_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list_all = list(subj_dict.keys())
    random.shuffle(subj_list_all)
    subj_list = subj_list_all[:subj_nbr]#['Brats17_CBICA_BFB_1_t2_unbiased.nii.gz'] #
    if subj_nbr == 1:
        subj_list = ['Brats17_TCIA_300_1_t2_unbiased.nii.gz']

    print(subj_list)

    # Init logging with Tensorboard
    writer = SummaryWriter(log_dir + name)

    if validation:
        subj_val_list = ['Brats17_CBICA_BHK_1_t2_unbiased.nii.gz'] # []
        # subj_val_list.append(subj_list_all[subj_nbr])
        print('validation subject', subj_val_list)
        writer_valid = SummaryWriter(log_dir + 'valid_' + name)

    slices = []
    for subj in subj_list:  # Iterate every subject
        slices.extend(subj_dict[subj])  # Slices for each subject

    # Load data
    subj_dataset = brats_dataset_subj(data_path, 'train', img_size, slices, use_aug=True)
    subj_loader = data.DataLoader(subj_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    print('Subject ', subj, ' Number of Slices: ', subj_dataset.size)

    for ep in range(epochs):
        #random.shuffle(subj_list)

        optimizer.zero_grad() # not needed
        for batch_idx, (scan, seg, mask) in enumerate(subj_loader):
            # Metrics init
            y_pred = []
            y_true = []

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
            mask = mask.squeeze(1)

            restored_batch, loss = train_run_map_NN(scan, decoded_mu, net, vae_model, riter, K_actf, step_rate,
                                                    device, writer, seg, mask)

            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('Loss', loss)

            seg = seg.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
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
            if not all(element==0 for element in y_true):
                AUC = roc_auc_score(y_true, y_pred)

            print('AUC : ', AUC)
            writer.add_scalar('AUC:', AUC, ep)
            writer.flush()

        if ep % log_freq == 0:
            # Save model
            path = '/scratch_net/biwidl214/jonatank/logs/restore/' + name + str(ep) + '.pth'
            torch.save(net, path)

            ## VALIDATION
            if validation:
                y_pred_valid = []
                y_true_valid = []
                slices = []
                tot_loss = 0

                for subj in subj_val_list:  # Iterate every subject
                    slices.extend(subj_dict[subj])  # Slices for each subject CHANGE

                # Load data
                valid_subj_dataset = brats_dataset_subj(data_path, 'train', img_size, slices, use_aug=False)
                valid_subj_loader = data.DataLoader(valid_subj_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
                print('Subject ', subj, ' Number of Slices: ', valid_subj_dataset.size)

                for batch_idx, (scan, seg, mask) in enumerate(valid_subj_loader):
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
                    mask = mask.squeeze(1)

                    restored_batch, loss = train_run_map_NN(scan, decoded_mu, net, vae_model, riter, K_actf,
                                                            step_rate, device, writer_valid, seg, mask,
                                                            train=False, log=bool(batch_idx % 2))

                    tot_loss += loss

                    seg = seg.cpu().detach().numpy()
                    mask = mask.cpu().detach().numpy()
                    # Predicted abnormalty is difference between restored and original batch
                    error_batch = np.zeros([scan.size()[0], original_size, original_size])
                    restored_batch_resized = np.zeros([scan.size()[0], original_size, original_size])

                    for idx in range(scan.size()[0]):  # Iterate trough for resize
                        error_batch[idx] = resize(abs(scan[idx] - restored_batch[idx]).cpu().detach().numpy(),
                                                  (200, 200))
                        restored_batch_resized[idx] = resize(restored_batch[idx].cpu().detach().numpy(), (200, 200))

                    # Remove preds and seg outside mask and flatten
                    mask = resize(mask, (scan.size()[0], original_size, original_size))
                    seg = resize(seg, (scan.size()[0], original_size, original_size))

                    error_batch_m = error_batch[mask > 0].ravel()
                    seg_m = seg[mask > 0].ravel().astype(bool)

                    # AUC
                    y_pred_valid.extend(error_batch_m.tolist())
                    y_true_valid.extend(seg_m.tolist())

                writer_valid.add_scalar('Loss', tot_loss/(batch_idx+1))
                AUC = roc_auc_score(y_true_valid, y_pred_valid)
                print('AUC Valid: ', AUC)
                writer_valid.add_scalar('AUC:', AUC, ep)
                writer_valid.flush()
                ## Write to tensorboard
                #writer_valid.add_image('Batch of Scan', scan.unsqueeze(1)[:16], batch_idx, dataformats='NCHW')
                #writer_valid.add_image('Batch of Restored', normalize_tensor(np.expand_dims(restored_batch_resized, axis=1)[:16]),
                #                 batch_idx, dataformats='NCHW')
                #writer_valid.add_image('Batch of Diff Restored Scan', normalize_tensor(np.expand_dims(error_batch, axis=1)[:16]),
                #                 batch_idx, dataformats='NCHW')
                #writer_valid.add_image('Batch of Ground truth', np.expand_dims(seg, axis=1)[:16], batch_idx, dataformats='NCHW')
                #

