__author__ = 'jonatank'
import sys
sys.path.insert(0, '/scratch_net/biwidl214/jonatank/code_home/restor_MAP/')

import torch
import torch.utils.data as data

from datasets import brats_dataset_subj
import numpy as np
import argparse
import yaml
import pickle
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=0)
    parser.add_argument("--config", required=True, help="path to config")

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    model_name = opt.model_name

    lr_rate = float(config['lr_rate'])
    data_path = config['path']
    epochs = config['epochs']
    batch_size = config["batch_size"]
    img_size = config["spatial_size"]
    log_freq = config['log_freq']
    log_dir = config['log_dir']
    log_model = config['model_dir']

    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    #test_dataset = brats_dataset(path, 'test', 128, prop_subjects=1)
    #test_data_loader  = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    #print('TEST loaded: Slices: ' + str(len(test_data_loader.dataset)) + ' Subjects: ' + str(int(50)))

    path_model = log_model + model_name + '.pth'
    model = torch.load(path_model, map_location=torch.device(device))
    model.eval()

    subj_dice = []
    f = open(data_path + 'subj_t2_test_dict.pkl', 'rb')
    subj_dict = pickle.load(f)
    f.close()

    subj_list = list(subj_dict.keys())

    with torch.no_grad():
        pred_list = []
        seg_list = []

        for subj in subj_list:
            TP = 0
            FN = 0
            FP = 0

            slices = subj_dict[subj]

            # Load data
            subj_dataset = brats_dataset_subj(data_path, 'test', img_size, slices)  # Change rand_subj to True
            subj_loader = data.DataLoader(subj_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
            #print('Subject ', subj, ' Number of Slices: ', subj_dataset.size)


            for batch_idx, (scan, seg, mask) in enumerate(subj_loader): #tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc='Evaluate'):
                scan = scan.to(device)
                pred_seg = model(scan.float())

                pred_seg = pred_seg.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
                seg = seg.cpu().detach().numpy()

                pred_m = pred_seg[mask > 0].ravel()
                seg_m = seg[mask > 0].ravel()

                pred_list.extend(pred_m)
                seg_list.extend(seg_m)

                pred_m = np.rint(pred_m)

                TP += np.sum(seg_m[pred_m == 1])
                FN += np.sum(seg_m[pred_m == 0])
                FP += np.sum(pred_m[seg_m == 0])

            dice = (2 * TP) / (2 * TP + FN + FP)

            subj_dice.append(dice)

            print(subj, ' DICE: ', dice)

        AUC = roc_auc_score(seg_list, pred_list)
        print('AUC', AUC)
        print('mean: ', np.mean(np.array(subj_dice), axis=0), ' std: ', np.std(np.array(subj_dice), axis=0))