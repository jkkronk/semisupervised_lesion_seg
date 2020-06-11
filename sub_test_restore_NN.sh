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
python -u restore_MAP_NN.py --name test_1subj_3lr_10steps_100K_2fch_BCE_4_ --config conf/conf_nn.yaml --netname 1subj_3lr_10steps_100K_2fch_BCE_4_300 --fprate 0 --subj '['Brats17_TCIA_208_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_1subj_3lr_10steps_100K_2fch_BCE_4_h --config conf/conf_nn.yaml --netname 1subj_3lr_10steps_100K_2fch_BCE_4_300 --fprate 0.005 --subj '['Brats17_TCIA_208_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_1subj_3lr_10steps_100K_2fch_BCE_5_ --config conf/conf_nn.yaml --netname 1subj_3lr_10steps_100K_2fch_BCE_5_300 --fprate 0 --subj '['Brats17_TCIA_192_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_1subj_3lr_10steps_100K_2fch_BCE_5_h --config conf/conf_nn.yaml --netname 1subj_3lr_10steps_100K_2fch_BCE_5_300 --fprate 0.005 --subj '['Brats17_TCIA_192_1_t2_unbiased.nii.gz']'
