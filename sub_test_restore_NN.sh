#!/bin/bash
#$ -o qsub_output
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=40G
#$ -q gpu.24h.q
source /scratch_net/biwidl214/jonatank/anaconda3/etc/profile.d/conda.sh
conda activate JKMT
python -u restore_MAP_NN.py --name test_10subj_3lr_10steps_100K_2fch_BCE_1_ --config conf/conf_nn.yaml --netname 10subj_3lr_10steps_100K_2fch_BCE_1_100 --fprate 0 --subj '['Brats17_TCIA_640_1_t2_unbiased.nii.gz', 'Brats17_TCIA_608_1_t2_unbiased.nii.gz', 'Brats17_TCIA_620_1_t2_unbiased.nii.gz', 'Brats17_TCIA_149_1_t2_unbiased.nii.gz', 'Brats17_TCIA_606_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AAG_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQO_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQR_1_t2_unbiased.nii.gz', 'Brats17_TCIA_351_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AVV_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_10subj_3lr_10steps_100K_2fch_BCE_1_h --config conf/conf_nn.yaml --netname 10subj_3lr_10steps_100K_2fch_BCE_1_100 --fprate 0.005 --subj '['Brats17_TCIA_208_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_10subj_3lr_10steps_100K_2fch_BCE_2_ --config conf/conf_nn.yaml --netname 10subj_3lr_10steps_100K_2fch_BCE_2_100 --fprate 0 --subj '['Brats17_TCIA_451_1_t2_unbiased.nii.gz', 'Brats17_TCIA_328_1_t2_unbiased.nii.gz', 'Brats17_TCIA_150_1_t2_unbiased.nii.gz', 'Brats17_2013_11_1_t2_unbiased.nii.gz', 'Brats17_TCIA_319_1_t2_unbiased.nii.gz', 'Brats17_TCIA_111_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AVG_1_t2_unbiased.nii.gz', 'Brats17_TCIA_254_1_t2_unbiased.nii.gz', 'Brats17_TCIA_274_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AYA_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_10subj_3lr_10steps_100K_2fch_BCE_2_h --config conf/conf_nn.yaml --netname 10subj_3lr_10steps_100K_2fch_BCE_2_100 --fprate 0.005 --subj '['Brats17_TCIA_192_1_t2_unbiased.nii.gz']'
