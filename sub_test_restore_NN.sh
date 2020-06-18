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
python -u restore_MAP_NN.py --name test_1subj_1e1_10steps_2fch2sh_DCS_lessaug_pelbo_mask_50h --config conf/conf_nn.yaml --netname 1subj_1e1_10steps_2fch2sh_DCS_lessaug_pelbo_mask_50 --fprate 0.005 --subj '['Brats17_TCIA_141_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_1subj_1e1_10steps_2fch2sh_DCS_lessaug_pelbo_mask_100h --config conf/conf_nn.yaml --netname 1subj_1e1_10steps_2fch2sh_DCS_lessaug_pelbo_mask_100 --fprate 0.005 --subj '['Brats17_TCIA_141_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_1subj_1e1_10steps_2fch2sh_DCS_lessaug_pelbo_mask_150h --config conf/conf_nn.yaml --netname 1subj_1e1_10steps_2fch2sh_DCS_lessaug_pelbo_mask_150 --fprate 0.005 --subj '['Brats17_TCIA_141_1_t2_unbiased.nii.gz']'
