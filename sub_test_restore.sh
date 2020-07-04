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
python -u restore_MAP_NN.py --name test_1subj_1e1_10steps_4fch3sh_iterAug_EX100_3_150 --config conf/conf_nn.yaml --netname 1subj_1e1_10steps_4fch3sh_iterAug_EX100_3_150 --fprate 0 --subj '['Brats17_TCIA_425_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_1subj_1e1_10steps_4fch3sh_iterAug_EX100_1_200 --config conf/conf_nn.yaml --netname 1subj_1e1_10steps_4fch3sh_iterAug_EX100_1_200 --fprate 0 --subj '['Brats17_TCIA_372_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_1subj_1e1_10steps_4fch3sh_iterAug_EX100_2_200 --config conf/conf_nn.yaml --netname 1subj_1e1_10steps_4fch3sh_iterAug_EX100_2_200 --fprate 0 --subj '['Brats17_TCIA_231_1_t2_unbiased.nii.gz']'


