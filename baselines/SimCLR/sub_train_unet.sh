#!/bin/bash
#$ -o qsub_output
#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=60G
#$ -q gpu.48h.q
source /scratch_net/biwidl214/jonatank/anaconda3/etc/profile.d/conda.sh
conda activate JKMT
python -u train_unet.py --model_name 200subj_0_simclr_ --config conf.yaml --aug 1 --subjs 200 --simCLR /scratch_net/biwidl214/jonatank/logs/simclr/1/checkpoints/model.pth






