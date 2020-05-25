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
python -u train_restore_MAP_NN.py --name 1subj_1K_4lr_15steps --config conf/conf_nn.yaml --subjs 2 --K_actf 1

python -u train_restore_MAP_NN.py --name 1subj_100K_4lr_15steps --config conf/conf_nn.yaml --subjs 2 --K_actf 100

python -u train_restore_MAP_NN.py --name 1subj_10000K_4lr_15steps --config conf/conf_nn.yaml --subjs 2 --K_actf 10000

python -u train_restore_MAP_NN.py --name 1subj_1000000K_4lr_15steps --config conf/conf_nn.yaml --subjs 2 --K_actf 1000000



