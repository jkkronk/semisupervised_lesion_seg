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
from restoration import train_run_map_GNN, train_run_map_NN, train_run_map_CNN
from models.shallow_UNET import shallow_UNet
from models.covnet import ConvNet
from datasets import brats_dataset_subj, camcan_dataset

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
    net = shallow_UNet(name, 3, 1, 2).to(device)
    #net = ConvNet(name, 2, 1, 32).to(device)
    #net = UNet(name, 2, 1, 4).to(device)

    #path = '/scratch_net/biwidl214/jonatank/logs/restore/1subj_3lr_1steps_32fch_2MSEloss_200.pth'
    #net = torch.load(path, map_location=torch.device(device))

    optimizer = optim.Adam(net.parameters(), lr=lr_rate)

    # Load list of subjects
    # Load data
    path = '/scratch_net/biwidl214/jonatank/data/dataset_abnormal/new/camcan/'
    validation_dataset = camcan_dataset(path, False, img_size, data_aug=1)
    valid_data_loader = data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Train data loaded')

    # Init logging with Tensorboard
    writer_train = SummaryWriter(log_dir + name + 'train')

    for ep in range(epochs):
        print(ep)
        for batch_idx, (scan, mask) in enumerate(valid_data_loader):  # tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):

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
            seg = torch.zeros(scan.shape)
            mask = mask.squeeze(1)

            restored_batch, loss = train_run_map_GNN(scan, decoded_mu, net, vae_model, 1, step_rate,
                                                    device, writer_train, seg, mask, healthy=True)

            optimizer.step()
            optimizer.zero_grad()

            writer_train.add_scalar('Loss', loss, ep)
            print(loss)

        # Save model
        path = '/scratch_net/biwidl214/jonatank/logs/restore/' + name + str(ep) + '.pth'
        torch.save(net, path)