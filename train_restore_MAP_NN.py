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
    n_latent_samples = config['latent_samples']
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

    train_slices = [559, 623, 19401, 19469, 600, 10202, 19506, 10107, 546, 10092, 10097, 10223, 10123, 10193, 10206, 19399, 10180, 10140, 10175, 19463, 19427, 10096, 576, 10203, 10126, 10152, 598, 10222, 19507, 628, 542, 19429, 10155, 672, 19422, 10210, 10166, 10176, 10191, 10172, 582, 638, 568, 631, 663, 19500, 10125, 19426, 536, 10122, 10095, 635, 19424, 538, 19516, 19483, 19437, 10156, 19445, 10213, 19502, 606, 566, 585, 10171, 674, 19470, 10124, 19423, 19433, 19468, 567, 556, 558, 10184, 547, 19442, 578, 10154, 10112, 661, 594, 19504, 10151, 590, 10118, 19459, 10093, 19414, 19418, 617, 19467, 10192, 592, 10129, 19412, 19435, 10127, 10098, 10197, 570, 548, 19503, 10214, 19439, 19480, 19397, 591, 651, 19415, 19478, 10091, 561, 641, 572, 19453, 10103, 19416, 655, 19513, 19472, 664, 19434, 574, 19406, 10215, 10137, 653, 19438, 544, 10130, 622, 643, 539, 19417, 626, 662, 10209, 612, 19395, 10187, 19473, 10207, 10190, 10090, 10221, 10121, 10128, 19496, 670, 10218, 10160, 603, 624, 646, 19490, 613, 637, 10115, 19413, 19521, 640, 10220, 608, 10201, 10185, 669, 10142, 10163, 580, 10196, 550, 629, 10139, 540, 19515, 610, 19407, 19488, 654, 611, 10164, 602, 10087, 10134, 537, 19430, 19451, 10132, 19484, 597, 583, 10104, 650, 552, 10181, 19519, 555, 19494, 19428, 10131, 10204, 19405, 19476, 19475, 19393, 19420, 10116, 10216, 19425, 19499, 10136, 10094, 19465, 675, 596, 10113, 19523, 19409, 10183, 632, 19419, 557, 10182, 607, 615, 19495, 19431, 19486, 584, 19447, 10158, 581, 549, 551, 614, 639, 19482, 10170, 10157, 19501, 656, 19403, 19485, 19456, 10146, 19489, 19511, 19446, 19421, 10111, 19440, 10219, 601, 10167, 10141, 10100, 673, 10162, 19398, 19527, 627, 658, 587, 19400, 660, 19522, 10150, 586, 657, 647, 10194, 619, 10174, 19528, 10195, 19462, 671, 630, 667, 19493, 19509, 19526, 10148, 621, 19461, 564, 10211, 19432, 19452, 10200, 577, 565, 10145, 19514, 10138, 665, 19466, 636, 659, 19520, 579, 19492, 609, 19512, 19411, 10099, 10165, 19450, 605, 10212, 10147, 10101, 571, 19471, 10143, 633, 10110, 19518, 10135, 10108, 10205, 573, 19464, 10173, 10089, 10105, 652, 10189, 19392, 554, 10133, 19449, 666, 616, 19436, 575, 10102, 588, 10153, 19444, 19410, 10217, 10109, 10106, 10120, 10088, 19443, 562, 543, 545, 10179]

    valid_slices = [553, 560, 19404, 541, 634, 10117, 648, 10159, 19458, 19474, 10149, 10186, 618, 19402, 10114, 644, 19491, 19391, 19441, 19455, 19477, 642, 19510, 19408, 620, 569, 668, 19517, 593, 10177, 10208, 19498, 19396, 10198, 599, 10178, 10199, 645, 604, 10119, 19481, 19525, 10144, 19508, 595, 649, 10161, 19487, 19457, 19524, 589, 19394, 19448, 19454, 10169, 625, 19497, 19505, 19460, 10168, 19479, 10188, 563]

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

            if K_actf == 0:
                restored_batch, loss = train_run_map_implicit(scan, decoded_mu, net, vae_model, riter, step_size,
                                                              device, writer, seg, mask, K_actf=K_actf)
            else:
                restored_batch, loss = train_run_map_explicit(scan, decoded_mu, net, vae_model, riter, step_size,
                                                              device, writer, seg, mask, K_actf=K_actf)

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
