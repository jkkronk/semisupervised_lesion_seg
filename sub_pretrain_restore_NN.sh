#!/bin/bash
#$ -o qsub_output
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=30G
#$ -q gpu.24h.q
source /scratch_net/biwidl214/jonatank/anaconda3/etc/profile.d/conda.sh
conda activate JKMT
python -u pretrain_restore_MAP_NN.py --name 1subj_1e1_1steps_2fch_2MSEloss_pretrain_aug_mask --config conf/conf_nn.yaml --subjs 1

