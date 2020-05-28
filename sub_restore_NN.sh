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
python -u restore_MAP_NN.py --name test_2subj_1K_4lr_15steps_norm --config conf/conf_nn.yaml --netname 2subj_1K_4lr_15steps_norm150 --fprate 0

python -u restore_MAP_NN.py --name test_2subj_1K_4lr_15steps_norm_healthy --config conf/conf_nn.yaml --netname 2subj_1K_4lr_15steps_norm150 --fprate 0.1

python -u restore_MAP_NN.py --name test_2subj_100K_4lr_15steps_norm --config conf/conf_nn.yaml --netname 2subj_100K_4lr_15steps_norm150 --fprate 0

python -u restore_MAP_NN.py --name test_2subj_1000K_4lr_15steps_norm --config conf/conf_nn.yaml --netname 2subj_1000K_4lr_15steps_norm150 --fprate 0

python -u restore_MAP_NN.py --name test_2subj_10000K_4lr_15steps_norm --config conf/conf_nn.yaml --netname 2subj_10000K_4lr_15steps_norm150 --fprate 0
