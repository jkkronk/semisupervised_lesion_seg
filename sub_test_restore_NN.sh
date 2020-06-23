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
python -u restore_MAP_NN.py --name test_3subj_1e1_10steps_2fch3sh_saug_0_ --config conf/conf_nn.yaml --netname 3subj_1e1_10steps_2fch3sh_saug_0_100 --fprate 0 --subj '['Brats17_TCIA_654_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQD_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AWG_1_t2_unbiased.nii.gz']'
