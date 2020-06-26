__author__ = 'jonatank'
from torch.utils.data import Dataset
from torchvision import transforms

import h5py
import torch
import numpy as np
from skimage.transform import resize
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import time
import imgaug as ia

class camcan_dataset(Dataset):
    def __init__(self, data_path, train, img_size, data_aug=0):
        self.img_size = img_size
        self.aug = data_aug
        path = (data_path + 'camcan_t2_train_set_4.hdf5') if train else (data_path + 'camcan_t2_val_set_4.hdf5')

        self.data = h5py.File(path, 'r')
        # Set size of dataset
        self.size = len(self.data['Scan'])

    def transform(self, img):
        # Function for data augmentation
        # 1) Affine Augmentations: Rotation (-15 to +15 degrees), Scaling, Flipping.
        # 2) Elastic deformations
        # 3) Intensity augmentations

        ia.seed(int(time.time())) # Seed for random augmentations

        # Needed for iaa
        img = (img*255).astype('uint8')

        if self.aug: # Augmentation only performed on train set
            img = np.expand_dims(img, axis=0)

            seq_all = iaa.Sequential([
                iaa.Fliplr(0.5), # Horizontal flips
                iaa.ElasticTransformation(alpha=(0.0, 20.0), sigma=5.0),  # Elastic
                iaa.blur.AverageBlur(k=(0, 3)),  # Gausian blur
                iaa.LinearContrast((0.8, 1.2)),  # Contrast
                iaa.Multiply((0.8, 1.2), per_channel=1)  # Intensity
            ], random_order=True)

            images_aug = seq_all(images=img) # Intensity and contrast only on input image

            img = np.squeeze(images_aug, axis=0)

        flip_tensor_trans = transforms.Compose([
            transforms.ToTensor()
        ])

        return flip_tensor_trans(img)

    def __getitem__(self, index):
        data_img = self.data['Scan'][index].reshape(200,200)

        # Resize Images to network
        data_img = resize(data_img, (self.img_size, self.img_size))

        # Expand to data with channel [1,128,128]
        data_img = np.expand_dims(data_img, axis=-1)

        mask = torch.zeros(data_img.shape)
        mask[data_img > 0] = 1

        img_trans = self.transform(data_img)

        return img_trans, mask.squeeze(-1)

    def __len__(self):
        return self.size

class brats_dataset_subj(Dataset):
    def __init__(self, data_path, dataset, img_size, slices, use_aug=False):
        self.img_size = img_size
        self.slices = slices
        self.dataset = dataset
        self.aug = use_aug

        # Open datasets
        if self.dataset == 'train':
            self.train = True
            print('Loading train set for subj')
            self.path = (data_path + 'brats17_t2_train.hdf5')
        elif self.dataset == 'valid':
            self.train = False
            print('Loading validation set for subj')
            self.path = (data_path + 'brats17_t2_val.hdf5')
        elif self.dataset == 'test':
            self.train = False
            print('Loading test set for subj')
            self.path = (data_path + 'brats17_t2_test.hdf5')
        else:
            print('No set named ' + set)
            exit()

        # Get subject list
        self.size = len(slices)

        # Load hdf5 file
        with h5py.File(self.path, 'r') as f:
            d = f

            # torch first saves this numpy array as a regular tensor and share_memory_() then copies it again to
            # a shared memory location. Therefore at least twice the size of the dataset / numpy matrix is needed
            # for memory.

            # Init data arrays
            self.data_img = np.zeros((self.size, 200, 200))
            self.seg_img = np.zeros((self.size, 200, 200), dtype='bool')

            # Iterate slices and place in arrays
            for idx, id_slice in enumerate(slices):
                self.data_img[idx] = torch.from_numpy(d.get('Scan')[id_slice].reshape(200, 200)).share_memory_()
                #self.data['Scan'][id_slice].reshape(200, 200)
                self.seg_img[idx] = torch.from_numpy(d.get('Seg')[id_slice].reshape(200, 200).astype(np.bool)).share_memory_()
                #self.data['Seg'][id_slice].reshape(200, 200)

            f.close()

    def transform(self, img, seg):
        # Function for data augmentation
        # 1) Affine Augmentations: Rotation (-15 to +15 degrees), Scaling, Flipping.
        # 2) Elastic deformations
        # 3) Intensity augmentations

        ia.seed(int(time.time()))  # Seed for random augmentations

        # Needed for iaa
        img = (img * 255).astype('uint8')
        seg = (seg).astype('uint8')

        if self.aug:  # Augmentation only performed on train set
            img = np.expand_dims(img, axis=0)
            segmap = SegmentationMapsOnImage(seg, shape=img.shape[1:])  # Create segmentation map

            seq_all = iaa.Sequential([
                iaa.Fliplr(0.5),  # Horizontal flips
                iaa.Affine(
                    scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
                    translate_percent={"x": (0, 0), "y": (0, 0)},
                    rotate=(-15, 15),
                    shear=(0, 0)),  # Scaling, rotating
                iaa.ElasticTransformation(alpha=(0.0, 20.0), sigma=5.0)  # Elastic
            ], random_order=True)

            seq_img = iaa.Sequential([
                iaa.blur.AverageBlur(k=(0, 3)),  # Gausian blur
                iaa.LinearContrast((0.8, 1.2)),  # Contrast
                iaa.Multiply((0.8, 1.2), per_channel=1),  # Intensity
            ], random_order=True)

            img, seg = seq_all(images=img, segmentation_maps=segmap)  # Rest of augmentations

            mask = np.zeros(img.shape) # Create mask
            mask[img > 0] = 1

            img = seq_img(images=img)  # Intensity and contrast only on input image

            img = np.squeeze(img, axis=0)
            mask = np.squeeze(mask,axis=0)

            # Get segmentation map
            seg = seg.draw(size=img.shape)[0]
            seg = seg[:, :, 0]
            seg[seg > 0] = 1
        else:
            mask = np.zeros(img.shape)
            mask[img > 0] = 1

        # To PIL for Flip and ToTensor
        img_PIL = Image.fromarray(img)
        seg_PIL = Image.fromarray(seg * 255)
        mask_PIL = Image.fromarray(mask)

        flip_tensor_trans = transforms.Compose([
            transforms.RandomVerticalFlip(p=1),  # Flipped due to camcan
            transforms.ToTensor()
        ])

        return flip_tensor_trans(img_PIL), flip_tensor_trans(seg_PIL), flip_tensor_trans(mask_PIL)

    def __getitem__(self, index):
        # Resize Images to network
        img_data = resize(self.data_img[index], (self.img_size, self.img_size))
        seg_data = resize(self.seg_img[index], (self.img_size, self.img_size))

        # Set all segmented elements to 1
        seg_data[seg_data > 0] = 1

        img_trans, seg_trans, mask_trans = self.transform(img_data, seg_data)

        return img_trans, seg_trans, mask_trans

    def __len__(self):
        return self.size
