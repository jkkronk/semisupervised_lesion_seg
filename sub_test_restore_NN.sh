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
python -u restore_MAP_NN.py --name test_1subj_3e1_10steps_2fch2sh_muchaug_lr2_nomask_1_150 --config conf/conf_nn.yaml --netname 1subj_3e1_10steps_2fch2sh_muchaug_lr2_nomask_1_150 --fprate 0 --subj '['Brats17_TCIA_325_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_1subj_3e1_10steps_2fch2sh_muchaug_lr2_nomask_1__150_h --config conf/conf_nn.yaml --netname 1subj_3e1_10steps_2fch2sh_muchaug_lr2_nomask_1_150 --fprate 0.005 --subj '['Brats17_TCIA_325_1_t2_unbiased.nii.gz']'
