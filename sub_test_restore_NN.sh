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
python -u restore_MAP_NN.py --name test_10subj_3e1_10steps_4fch3sh_saug_2_ --config conf/conf_nn.yaml --netname 10subj_3e1_10steps_4fch3sh_saug_2_125 --fprate 0 --subj '['Brats17_TCIA_412_1_t2_unbiased.nii.gz', 'Brats17_TCIA_201_1_t2_unbiased.nii.gz', 'Brats17_2013_6_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AXW_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AXO_1_t2_unbiased.nii.gz', 'Brats17_TCIA_499_1_t2_unbiased.nii.gz', 'Brats17_TCIA_218_1_t2_unbiased.nii.gz', 'Brats17_TCIA_625_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ASW_1_t2_unbiased.nii.gz', 'Brats17_TCIA_479_1_t2_unbiased.nii.gz']'
