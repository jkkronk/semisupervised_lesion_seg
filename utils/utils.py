import torch
import numpy as np
import time
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def normalize_tensor(input_tens):
    i_max = input_tens.max()
    i_min = input_tens.min()
    input_tens = (input_tens-i_min)/(i_max-i_min)
    return input_tens

def normalize_tensor_N(input_tens, N):
    i_max = input_tens.max()
    i_min = input_tens.min()
    input_tens = (input_tens-i_min)/(i_max-i_min)
    input_tens = input_tens*(N/torch.mean(input_tens)) #e=2.718281
    return input_tens

class diceloss(torch.nn.Module):
    def init(self):
        super(diceloss, self).init()
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

def dice_loss(prediction, target):
    # Dice loss
    prediction = prediction[:, 0].contiguous().view(-1)
    target = target[:, 0].contiguous().view(-1)
    intersection = (prediction * target).sum()
    return 1 - ((2. * intersection + 1) / (prediction.sum() + target.sum() + 1))

def total_variation(images):
    """
    Edited from tensorflow implementation

    Calculate and return the total variation for one or more images.

    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input images. This measures how much noise is in the
    images.

    This implements the anisotropic 2-D version of the formula described here:
    https://en.wikipedia.org/wiki/Total_variation_denoising

    Args:
        images: 3-D Tensor of shape `[batch, height, width]`.
    Returns:
        The total variation of `images`.

        return a scalar float with the total variation for
        that image.
    """

    # The input is a single image with shape [batch, height, width].

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height and width by slicing.
    pixel_dif1 = images[:, 1:, :] - images[:, :-1, :]
    pixel_dif2 = images[:, :, 1:] - images[:, :, :-1]

    # Sum for all axis. (None is an alias for all axis.)

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = (
        torch.sum(torch.abs(pixel_dif1)) +
        torch.sum(torch.abs(pixel_dif2)))

    return tot_var

def composed_tranforms(img_tensor, seg_tensor):
    # Function for data augmentation
    # 1) Affine Augmentations: Rotation (-15 to +15 degrees), Scaling, Flipping.
    # 2) Elastic deformations
    # 3) Intensity augmentations

    ia.seed(int(time.time()))  # Seed for random augmentations
    N, C, H, W = img_tensor.shape
    mask_tensor = torch.zeros((N,H,W))

    # Needed for iaa
    for i in range(img_tensor.shape[0]):
        img = img_tensor[i].detach().cpu().numpy().transpose((1, 2, 0))

        seg = seg_tensor[i].detach().cpu().numpy().astype('bool')

        segmap = SegmentationMapsOnImage(seg, shape=img.shape)  # Create segmentation map

        seq_all = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontal flips
            iaa.Flipud(0.5),
            iaa.Affine(
                scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
                translate_percent={"x": (0, 0), "y": (0, 0)},
                rotate=(-90, 90),
                shear=(0, 0)),  # Scaling, rotating
            iaa.ElasticTransformation(alpha=(0.0, 0.50), sigma=6.0)  # Elastic
        ], random_order=True)

        seq_img = iaa.Sequential([
            iaa.blur.AverageBlur(k=(0, 4)),  # Gausian blur
            iaa.LinearContrast((0.6, 1.4)),  # Contrast
            iaa.Multiply((0.7, 1.3), per_channel=1),  # Intensity
        ], random_order=True)

        img, seg = seq_all(image=img, segmentation_maps=segmap)  # Rest of augmentations

        # Fix mask array before intensity augmentation
        mask_aug = np.zeros(img.shape)
        mask_aug[img[:, :, 0] > 0] = 1
        mask_aug = mask_aug[:,:,0]

        img = seq_img(image=img)  # Intensity and contrast only on input image

        # Fix segmentation array
        seg = seg.draw(size=img.shape)[0]
        seg = seg[:, :, 0]
        seg[seg > 0] = 1

        # To PIL for Flip and ToTensor
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask_aug = torch.from_numpy(mask_aug)
        seg = torch.from_numpy(seg)

        img_tensor[i] = img
        mask_tensor[i] = mask_aug
        seg_tensor[i] = seg

    return img_tensor, seg_tensor, mask_tensor