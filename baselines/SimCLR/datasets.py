__author__ = 'jonatank'
from torch.utils.data import Dataset
from torchvision import transforms

import h5py
import torch
import numpy as np
from skimage.transform import resize
import pickle
import random
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import time
import imgaug as ia
from matplotlib import cm

class camcan_dataset(Dataset):
    def __init__(self, data_path, train, img_size, data_aug):

        self.img_size = img_size
        self.transform = data_aug

        path = (data_path + 'camcan_t2_train_set_4.hdf5') if train else (data_path + 'camcan_t2_val_set_4.hdf5')

        self.data = h5py.File(path, 'r')
        # Set size of dataset
        self.size = len(self.data['Scan'])

    def __getitem__(self, index):
        data_img = self.data['Scan'][index].reshape(200,200)

        # Resize Images to network
        data_img = resize(data_img, (self.img_size, self.img_size))

        # Expand to data with channel [1,128,128]
        data_img = np.expand_dims(data_img, axis=-1)

        mask = torch.zeros(data_img.shape)
        mask[data_img > 0] = 1

        data_img = np.squeeze(data_img, axis=-1)

        PIL_image = Image.fromarray(np.uint8(data_img*255)).convert('RGB')

        img_trans = self.transform(PIL_image)

        img_ret = (torch.sum(img_trans[0], dim=0).unsqueeze(0), torch.sum(img_trans[1], dim=0).unsqueeze(0))

        return img_ret

    def __len__(self):
        return self.size
