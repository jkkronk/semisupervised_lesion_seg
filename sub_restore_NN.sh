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
python -u restore_MAP_NN.py --name test_1subj_4lr_20steps_muchAug --config conf/conf_nn.yaml --netname 1subj_4lr_20steps_muchAug150 --fprate 0.005 --subj '['Brats17_TCIA_300_1_t2_unbiased.nii.gz']'


