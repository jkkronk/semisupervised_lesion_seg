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

python -u restore_MAP_NN.py --name test_30subj_1e1_10steps_4fch3sh_iterAug_EX100_1_ --config conf/conf_nn.yaml --netname 30subj_1e1_10steps_4fch3sh_iterAug_EX100_1_20 --fprate 0 --subj '['Brats17_TCIA_141_1_t2_unbiased.nii.gz', 'Brats17_TCIA_633_1_t2_unbiased.nii.gz', 'Brats17_TCIA_109_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ANZ_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ALU_1_t2_unbiased.nii.gz']'
