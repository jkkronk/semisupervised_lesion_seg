# CODE FROM XIAORAN CHEN https://github.com/aubreychen9012/AutoEncoder_AnomalyDetection

import h5py
import random
import numpy as np
from skimage.transform import resize
import torch

from utils import losses

#sys.path.append("/scratch_net/bmicdl01/chenx/PycharmProjects/refine_vae")
#from preprocess.preprocess import *

def determine_threshold(phi, fprate):
    phi = np.asarray(phi)
    """
    determines the lowest threshold on Phi that provides at max FP rate on the Phi values.
    all the samples need to be controls for this function
    """
    nums = len(phi)
    #numf = phi.shape[1]

    def func(threshold):
        phi_ = phi > threshold
        fprate_ = np.sum(phi_) / np.float(nums)
        return np.sqrt((fprate - fprate_) ** 2)
    return gss(func, phi.min(), phi.mean(), phi.max(), tau=1e-8)


def gss(f, a, b, c, tau=1e-3):
    """
    Python recursive version of Golden Section Search algorithm

    tau is the tolerance for the minimal value of function f
    b is any number between the interval a and c
    """

    goldenRatio = (1 + 5 ** 0.5) / 2
    if c - b > b - a:
        x = b + (2 - goldenRatio) * (c - b)
    else:
        x = b - (2 - goldenRatio) * (b - a)
    if abs(c - a) < tau * (abs(b) + abs(x)): return (c + a) / 2
    if f(x) < f(b):
        if c - b > b - a:
            return gss(f, b, x, c, tau)
        return gss(f, a, x, b, tau)
    else:
        if c - b > b - a:
            return gss(f, a, b, x, tau)
        return gss(f, x, b, c, tau)

def mad_score(x, med):
    #minval = (x-mean).min()
    score = np.median(losses.l2loss_np(x, med), axis=-1, keepdims=True)
    return score

def modified_z_score(x, x_hat):
    median_x = np.median(x_hat, axis=-1, keepdims=True)
    MAD = np.median(np.abs(x_hat - median_x), axis=-1, keepdims=True)
    M = np.abs(x-median_x)/(MAD+1e-9)
    return M

def minibatches(inputs=None, targets=None, batch_size=None, allow_dynamic_batch_size=False, shuffle=False):
    """
    TENSORFLOW SOURCE CODE

    Generate a generator that input a group of example in numpy.array and
    their labels, return the examples and labels by the given batch size.

    Parameters
    ----------
    inputs : numpy.array
        The input features, every row is a example.
    targets : numpy.array
        The labels of inputs, every row is a example.
    batch_size : int
        The batch size.
    allow_dynamic_batch_size: boolean
        Allow the use of the last data batch in case the number of examples is not a multiple of batch_size, this may result in unexpected behaviour if other functions expect a fixed-sized batch-size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before return.
    Notes
    -----
    If you have two inputs and one label and want to shuffle them together, e.g. X1 (1000, 100), X2 (1000, 80) and Y (1000, 1), you can stack them together (`np.hstack((X1, X2))`)
    into (1000, 180) and feed to ``inputs``. After getting a batch, you can split it back into X1 and X2.

    """
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    # for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    # chulei: handling the case where the number of samples is not a multiple of batch_size, avoiding wasting samples
    for start_idx in range(0, len(inputs), batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len(inputs):
            if allow_dynamic_batch_size:
                end_idx = len(inputs)
            else:
                break
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        if (isinstance(inputs, list) or isinstance(targets, list)) and (shuffle ==True):
            # zsdonghao: for list indexing when shuffle==True
            yield [inputs[i] for i in excerpt], [targets[i] for i in excerpt]
        else:
            yield inputs[excerpt], targets[excerpt]

def compute_threshold(fprate, model, img_size, batch_size, n_latent_samples, device, n_random_sub = 100,
                      renormalized = False):
    fprate = fprate
    if renormalized:
        #n = "_renormalized"
        data = h5py.File('/scratch_net/biwidl214/jonatank/data/dataset_abnormal/new/camcan/camcan_t2_train_set.hdf5')
    else:
        data = h5py.File('/scratch_net/biwidl214/jonatank/data/dataset_abnormal/new/camcan/camcan_t2_train_set.hdf5')
    indices = random.sample(range(len(data['Scan']))[::batch_size], n_random_sub)

    image_size = img_size
    image_original_size = 200
    batch_size = batch_size
    dif = []
    dif_vae = []
    dif_vae_rel = []
    dif_prob = []
    dif_naive = []
    num = 0
    n_latent_samples = n_latent_samples
    for ind in indices:
        print(num, ind)
        res = data['Scan'][ind:ind + batch_size]
        res = res.reshape(-1, image_original_size, image_original_size)
        mask = data['Mask'][ind:ind + batch_size]
        mask = mask.reshape(-1, image_original_size, image_original_size)

        dim_res = res.shape
        image_original_size = res.shape[1]
        res_minval = res.min()

        if dim_res[0] % batch_size:
            dim_res_expand = batch_size - (dim_res[0] % batch_size)
            res_expand = np.zeros((dim_res_expand, dim_res[1], dim_res[2])) + res_minval
            res_exp = np.append(res, res_expand, axis=0)
            mask_exp = np.append(mask, np.zeros((dim_res_expand, dim_res[1], dim_res[2])), axis=0)
        else:
            res_exp = res
            mask_exp = mask

        res_exp = resize(res_exp, [batch_size, img_size, img_size])
        mask_exp = resize(mask_exp, [batch_size, img_size, img_size])

        cnt = 0
        predicted_residuals = []
        predicted_residuals_vae = []
        predicted_residuals_vae_relative = []
        prob_map = []

        for batch in minibatches(inputs=res_exp, targets=mask_exp,
                                            batch_size=batch_size, shuffle=False):
            b_images, _ = batch
            b_images = b_images[:, :, :, np.newaxis]
            b_images = torch.from_numpy(b_images).double().to(device)
            b_images = torch.squeeze(b_images, axis=3)
            b_images = torch.unsqueeze(b_images, axis=1)
            decoded = np.zeros((batch_size, n_latent_samples+1, image_size, image_size))
            for i in range(n_latent_samples):
                decoded_vae, _, _, res = model(b_images)
                #model.validate(b_images)
                #decoded_vae = model.out_mu_test
                decoded_vae_res = np.abs((b_images-decoded_vae).cpu().detach().numpy()) #np.abs(b_images - decoded_vae)
                decoded[:,i,:,:] = decoded_vae_res[:,0,:,:]

            # predicted model error
            decoded_res = res
            decoded[:,-1,:,:] = decoded_res.cpu().detach().numpy()[:,0,:,:]

            batch_median = np.median(decoded, axis=-1, keepdims=True)


            residuals_raw = np.abs((b_images - decoded_vae).cpu().detach().numpy())

            residuals = np.abs(residuals_raw - decoded_res.cpu().detach().numpy())

            predicted_residuals.extend(residuals)
            # raw
            predicted_residuals_vae.extend(residuals_raw)
            # signed

            # evaluate if predicted loss fits error distribution during test time
            residuals_mad_map = np.median(np.abs(decoded-batch_median), axis=1, keepdims=True)

            prob_map.extend(residuals_mad_map)
            cnt += 1

        predicted_residuals_vae = np.asarray(predicted_residuals_vae).reshape(res_exp.shape[0], 128, 128)
        predicted_residuals = np.asarray(predicted_residuals).reshape(res_exp.shape[0], 128, 128)

        predicted_residuals_vae = resize(predicted_residuals_vae[:dim_res[0]], [batch_size, 1, image_original_size, image_original_size])
        predicted_residuals = resize(predicted_residuals[:dim_res[0]], [batch_size, 1, image_original_size, image_original_size])

        prob_map = np.asarray(prob_map).reshape(res_exp.shape[0], 128, 128)
        prob_map = resize(prob_map[:dim_res[0]], [batch_size, 1, image_original_size, image_original_size])

        predicted_residuals_vae = np.squeeze(predicted_residuals_vae, axis=1)
        predicted_residuals = np.squeeze(predicted_residuals, axis=1)
        prob_map = np.squeeze(prob_map, axis=1)

        dif.extend(predicted_residuals[mask == 1])
        dif_vae.extend(predicted_residuals_vae[mask == 1])
        #dif_vae_rel.extend(predicted_residuals_vae_relative[mask == 1])
        dif_prob.extend(prob_map[mask == 1])
        #dif_naive.extend(res[mask == 1])
        num += 1

    thr_error = determine_threshold(dif_vae, fprate)
    thr_error_corr = determine_threshold(dif, fprate)
    thr_MAD = determine_threshold(dif_prob, fprate)

    return thr_error, thr_error_corr, thr_MAD
