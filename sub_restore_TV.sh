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
python -u restore_MAP_TV.py --name TV_old_10 --config conf/conf_TV.yaml --fprate 0.1

python -u restore_MAP_TV.py --name TV_old_01 --config conf/conf_TV.yaml --fprate 0.01

python -u restore_MAP_TV.py --name TV_old_05 --config conf/conf_TV.yaml --fprate 0.05



