#!/bin/bash
#$ -o qsub_output
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=50G
#$ -q gpu.24h.q
source /scratch_net/biwidl214/jonatank/anaconda3/etc/profile.d/conda.sh
conda activate JKMT
python -u train_restore_MAP_NN.py --name 10subj_1K_4lr_15steps_norm --config conf/conf_nn.yaml --subjs 11 --K_actf 1

python -u train_restore_MAP_NN.py --name 25subj_1K_4lr_15steps_norm --config conf/conf_nn.yaml --subjs 25 --K_actf 1

python -u train_restore_MAP_NN.py --name 50subj_1K_4lr_15steps_norm --config conf/conf_nn.yaml --subjs 50 --K_actf 1



