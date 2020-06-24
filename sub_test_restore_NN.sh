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
python -u restore_MAP_NN.py --name test_3subj_3e1_10steps_4fch3sh_saug_3_new_200 --config conf/conf_nn.yaml --netname 3subj_3e1_10steps_4fch3sh_saug_3_new_200 --fprate 0 --subj '['Brats17_TCIA_606_1_t2_unbiased.nii.gz', 'Brats17_TCIA_420_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQN_1_t2_unbiased.nii.gz']'