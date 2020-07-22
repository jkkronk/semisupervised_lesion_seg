# Guiding Unsupervised Image Restoration with few Annotated Subjects for Lesion Detection

Lesion detection is a critical task for medical image understanding. While the problem has been widely addressed in a supervised semantic segmentation manner, the problem clinically appears more similar to novelty detection with few or no annotations for the lesion. The reason is two-fold: 1) it is intuitively easier to collect large dataset from healthy individuals than that from a specific type of lesion individuals, 2) clinicians are generally interested in any abnormalities regardless of its type. This makes unsupervised methods more attractive solutions. Works such as AnoGAN and VAE with image restoration offer practical ways to localise lesions by training only on healthy data. However, for the same type of lesion, an obvious performance gap exists between unsupervised and supervised methods. In this work, we intend to provide supervision with a small number of lesion data to the unsupervised method with the aim to narrow the gap. The method is an extension of the unsupervised method of VAE with MAP-based image restoration. In more details, we train an U-Net on the few examples to predict the likelihood term and impose the supervision with annotated lesions such that the restoration only occurs for the lesion pixels. We train the unsupervised method on T2-weighted images of healthy individuals of Cam-CAN dataset and provide a small annotated dataset consisting of a few subjects from BraTS dataset, and test on an unseen subset of BraTS. With the addition of the few examples, the method shows an improvement over the unsupervised method while the gap with the supervised is narrowed but still exists.

This work was done at Computer vision lab (ETH) spring 2020. Feel free to send questions to jonatan@kronander.se

To train VAE: train_vae.py
To train Segmentation network: train_restore_MAP_NN.py
To test/perform restoration: restore_MAP_NN.py

Baselines methods are found in branch "baselines"
