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
python -u train_restore_MAP_NN.py --name 100subj_1e1_10steps_4fch3sh_iterAug_EX100_1_ --config conf/conf_nn.yaml --subjs 100 --K_actf 100
