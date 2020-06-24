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
from restoration import train_run_map_GGNN, train_run_map_NN, train_run_map_CNN
from models.shallow_UNET import shallow_UNet
from models.covnet import ConvNet
from datasets import brats_dataset_subj

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
    step_size = float(config['step_rate'])
    log_freq = config['log_freq']
    original_size = config['orig_size']
    log_dir = config['log_dir']
    n_latent_samples = 25
    epochs = config['epochs']

    validation = True

    print('Name: ', name, 'Lr_rate: ', lr_rate, ' Riter: ', riter, ' Step size: ', step_size)

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    # Load trained vae model
    vae_path = '/scratch_net/biwidl214/jonatank/logs/vae/'
    path = vae_path + model_name + '.pth'
    vae_model = torch.load(path, map_location=torch.device(device))
    vae_model.eval()

    # Create guiding net
    net = shallow_UNet(name, 2, 1, 4).to(device)
    #net = ConvNet(name, 2, 1, 32).to(device)
    #net = UNet(name, 2, 1, 4).to(device)

    #path = '/scratch_net/biwidl214/jonatank/logs/restore/1subj_1e1_1steps_2fch_2MSEloss_pretrain_aug_mask1.pth'
    #net = torch.load(path, map_location=torch.device(device))

    optimizer = optim.Adam(net.parameters(), lr=lr_rate)

    # Load list of subjects
    f = open(data_path + 'subj_t2_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list_all = list(subj_dict.keys())
    random.shuffle(subj_list_all)
    subj_list = subj_list_all[:subj_nbr]#['Brats17_CBICA_BFB_1_t2_unbiased.nii.gz'] #
    #if subj_nbr == 1:
    #    subj_list = ['Brats17_2013_14_1_t2_unbiased.nii.gz'] #5 Brats17_TCIA_451_1_t2_unbiased 4 Brats17_CBICA_AUN_1_t2_unbiased 3 Brats17_TCIA_105_1_t2_unbiased 2 Brats17_CBICA_AXW_1_t2_unbiased 1 Brats17_TCIA_241_1_t2_unbiased  0 Brats17_2013_14_1_t2_unbiased

    print(subj_list)

    slices = []
    for subj in subj_list:  # Iterate every subject
        slices.extend(subj_dict[subj])  # Slices for each subject

    if validation:
        random.shuffle(slices)
        train_slices = slices[:int((len(slices)*0.85))]
        valid_slices = slices[int((len(slices)*0.85)):]
        print("Train slices: ", train_slices)
        print("Validation slices: ", valid_slices)
        valid_writer = SummaryWriter(log_dir + "valid_" + name)

    #train_slices = [25346, 25359, 25365, 25335, 25243, 25342, 25250, 25296, 25309, 25350, 25288, 25319, 25378, 25276, 25265, 25264, 25324, 25337, 25271, 25366, 25300, 25244, 25260, 25245, 25339, 25375, 25311, 25313, 25290, 25357, 25343, 25347, 25322, 25256, 25328, 25340, 25331, 25374, 25345, 25285, 25323, 25316, 25349, 25252, 25351, 25364, 25317, 25330, 25239, 25240, 25314, 25332, 25280, 25344, 25301, 25286, 25363, 25302, 25274, 25315, 25255, 25368, 25321, 25292, 25373, 25241, 25293, 25238, 25270, 25253, 25305, 25320, 25371, 25272, 25353, 25247, 25333, 25304, 25254, 25278, 25279, 25299, 25277, 25294, 25369, 25297, 25251, 25361, 25283, 25303, 25336, 25248, 25356, 25258, 25263, 25257, 25275, 25259, 25307, 25282, 25318, 25327, 25370, 25354, 25325, 25358, 25376, 25262, 25269, 25291, 25266, 25237, 25329, 25362, 25312, 25242, 25341, 25355, 25360, 25268]
    #valid_slices = [25348, 25367, 25310, 25352, 25246, 25284, 25261, 25298, 25267, 25289, 25338, 25273, 25287, 25334, 25295, 25372, 25281, 25249, 25377, 25326, 25308, 25306]

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

            restored_batch, loss = train_run_map_GGNN(scan, decoded_mu, net, vae_model, riter, step_size,
                                                      device, writer, seg, mask, K_actf=K_actf, aug=True)

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

                restored_batch, loss = train_run_map_GGNN(scan, decoded_mu, net, vae_model, riter, step_size,
                                                          device, writer, seg, mask, K_actf=K_actf)

                valid_writer.add_scalar('Loss', loss)

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
                valid_y_pred.extend(error_batch_m.tolist())
                valid_y_true.extend(seg_m.tolist())
                if not all(element == 0 for element in valid_y_true):
                    AUC = roc_auc_score(valid_y_true, valid_y_pred)

                print('Valid AUC : ', AUC)
                valid_writer.add_scalar('AUC:', AUC, ep)
                valid_writer.flush()

        if ep % log_freq == 0:
            # Save model
            path = '/scratch_net/biwidl214/jonatank/logs/restore/' + name + str(ep) + '.pth'
            torch.save(net, path)
