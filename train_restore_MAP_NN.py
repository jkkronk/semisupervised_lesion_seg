import numpy as np
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
import pickle
import argparse
import yaml
import random
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from restoration import train_run_map_implicit, train_run_map_explicit
from models.shallow_UNET import shallow_UNet
from datasets import brats_dataset_subj

if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=0)
    parser.add_argument("--config", required=True, help="Path to config")
    parser.add_argument('--subjs', type=int, required=True, help="Number of subjects")
    parser.add_argument('--K_actf', type=int, default=1, help="Activation param: 0 = implicit approach")

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
    step_size = float(config['step_rate'])
    log_freq = config['log_freq']
    original_size = config['orig_size']
    log_dir = config['log_dir']
    n_latent_samples = config['latent_samples']
    epochs = config['epochs']

    print('Name: ', name, 'Lr_rate: ', lr_rate, ' Riter: ', riter, ' Step size: ', step_size, 'Kact:', K_actf)

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    # Load trained vae model
    vae_path = '/scratch_net/biwidl214/jonatank/logs/vae/'
    path = vae_path + model_name + '.pth'
    vae_model = torch.load(path, map_location=torch.device(device))
    vae_model.eval()

    # Create guiding net and init optimiser
    net = shallow_UNet(name, 2, 1, 4).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr_rate)

    # Load list of subjects
    f = open(data_path + 'subj_t2_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list_all = list(subj_dict.keys())
    random.shuffle(subj_list_all) # Shuffle subject list to pick random subset
    subj_list = subj_list_all[:subj_nbr]
    print(subj_list)

    train_list = subj_list[:int(0.85*subj_nbr)]
    train_slices = []
    for subj in subj_list:  # Iterate every subject
        train_slices.extend(subj_dict[subj])  # Slices for each subject

    # 15 % validation set
    random.shuffle(train_slices)
    valid_slices = train_slices[int(0.85*len(train_slices)):]
    train_slices = train_slices[:int(0.85*len(train_slices))]

    # Load data
    subj_dataset = brats_dataset_subj(data_path, 'train', img_size, train_slices, use_aug=False)
    subj_loader = data.DataLoader(subj_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    print('Subject ', subj, ' Number of Slices: ', subj_dataset.size)

    # Load data
    valid_subj_dataset = brats_dataset_subj(data_path, 'train', img_size, valid_slices, use_aug=False)
    valid_subj_loader = data.DataLoader(valid_subj_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    print('Subject ', subj, ' Number of Slices: ', valid_subj_dataset.size)

    # Init logging with Tensorboard
    writer = SummaryWriter(log_dir + name)
    valid_writer = SummaryWriter(log_dir + "valid_" + name)

    for ep in range(epochs):

        optimizer.zero_grad()
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

            # Restore
            if K_actf == 0:
                restored_batch, loss = train_run_map_implicit(scan, decoded_mu, net, vae_model, riter, step_size,
                                                              device, writer, seg, mask, K_actf=K_actf)
            else:
                restored_batch, loss = train_run_map_explicit(scan, decoded_mu, net, vae_model, riter, step_size,
                                                              device, writer, seg, mask, K_actf=K_actf)
            # Update segmentation network params
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('Loss', loss)

            seg = seg.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()

            # Predicted lesion is difference between restored and original
            error_batch = np.zeros([scan.size()[0],original_size,original_size])
            restored_batch_resized = np.zeros([scan.size()[0],original_size,original_size]) # For plotting restored

            for idx in range(scan.size()[0]): # Iterate for resize to original size
                error_batch[idx] = resize(abs(scan[idx] - restored_batch[idx]).cpu().detach().numpy(), (200,200))
                restored_batch_resized[idx] = resize(restored_batch[idx].cpu().detach().numpy(), (200,200))

            # Flatten and remove pred outside mask
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

        ## VALIDATION
        if validation and ep % 5 == 0:

            for batch_idx, (scan, seg, mask) in enumerate(valid_subj_loader):
                # Metrics init
                valid_y_pred = []
                valid_y_true = []

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

                # Restore
                if K_actf == 0:
                    restored_batch, loss = train_run_map_implicit(scan, decoded_mu, net, vae_model, riter, step_size,
                                                                  device, writer, seg, mask, aug=False,)
                else:
                    restored_batch, loss = train_run_map_explicit(scan, decoded_mu, net, vae_model, riter, step_size,
                                                                  device, writer, seg, mask, aug=False, K_actf=K_actf)
                optimizer.zero_grad()

                valid_writer.add_scalar('Loss', loss)

                seg = seg.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()

                # Predicted lesion is difference between restored and original
                error_batch = np.zeros([scan.size()[0], original_size, original_size])
                restored_batch_resized = np.zeros([scan.size()[0], original_size, original_size])

                for idx in range(scan.size()[0]):  # Iterate trough for resize
                    error_batch[idx] = resize(abs(scan[idx] - restored_batch[idx]).cpu().detach().numpy(),
                                              (200, 200))
                    restored_batch_resized[idx] = resize(restored_batch[idx].cpu().detach().numpy(), (200, 200))

                # Flatten and remove pred outside mask
                mask = resize(mask, (scan.size()[0], original_size, original_size))
                seg = resize(seg, (scan.size()[0], original_size, original_size))

                error_batch_m = error_batch[mask > 0].ravel()
                seg_m = seg[mask > 0].ravel().astype(bool)

                # AUC
                valid_y_pred.extend(error_batch_m.tolist())
                valid_y_true.extend(seg_m.tolist())
                if not all(element == 0 for element in valid_y_true):
                    AUC = roc_auc_score(valid_y_true, valid_y_pred)

                print('Valid AUC : ', AUC)
                valid_writer.add_scalar('AUC:', AUC, ep)
                valid_writer.flush()

        # Save model
        if ep % log_freq == 0:
            path = '/scratch_net/biwidl214/jonatank/logs/restore/' + name + str(ep) + '.pth'
            torch.save(net, path)
