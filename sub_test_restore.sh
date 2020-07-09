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

python -u restore_MAP_NN.py --name test_30subj_1e1_10steps_4fch3sh_iterAug_EX100_0_ --config conf/conf_nn.yaml --netname 30subj_1e1_10steps_4fch3sh_iterAug_EX100_0_20 --fprate 0 --subj '['Brats17_TCIA_629_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ANG_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ATX_1_t2_unbiased.nii.gz', 'Brats17_TCIA_321_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ASA_1_t2_unbiased.nii.gz', 'Brats17_TCIA_654_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ASK_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_30subj_1e1_10steps_4fch3sh_iterAug_EX100_2_ --config conf/conf_nn.yaml --netname 30subj_1e1_10steps_4fch3sh_iterAug_EX100_2_50 --fprate 0 --subj '['Brats17_TCIA_141_1_t2_unbiased.nii.gz', 'Brats17_CBICA_BFP_1_t2_unbiased.nii.gz', 'Brats17_TCIA_474_1_t2_unbiased.nii.gz', 'Brats17_TCIA_498_1_t2_unbiased.nii.gz', 'Brats17_TCIA_632_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AOH_1_t2_unbiased.nii.gz', 'Brats17_2013_2_1_t2_unbiased.nii.gz', 'Brats17_TCIA_276_1_t2_unbiased.nii.gz', 'Brats17_TCIA_430_1_t2_unbiased.nii.gz', 'Brats17_TCIA_300_1_t2_unbiased.nii.gz', 'Brats17_TCIA_328_1_t2_unbiased.nii.gz', 'Brats17_TCIA_471_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_30subj_1e1_10steps_4fch3sh_iterAug_EX100_3_ --config conf/conf_nn.yaml --netname 30subj_1e1_10steps_4fch3sh_iterAug_EX100_3_50 --fprate 0 --subj '['Brats17_TCIA_498_1_t2_unbiased.nii.gz', 'Brats17_TCIA_325_1_t2_unbiased.nii.gz', 'Brats17_2013_1_1_t2_unbiased.nii.gz', 'Brats17_2013_26_1_t2_unbiased.nii.gz', 'Brats17_TCIA_637_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AZH_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_30subj_1e1_10steps_4fch3sh_iterAug_EX100_4_ --config conf/conf_nn.yaml --netname 30subj_1e1_10steps_4fch3sh_iterAug_EX100_4_50 --fprate 0 --subj '['Brats17_TCIA_637_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQQ_1_t2_unbiased.nii.gz', 'Brats17_TCIA_343_1_t2_unbiased.nii.gz', 'Brats17_TCIA_605_1_t2_unbiased.nii.gz', 'Brats17_TCIA_165_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AAL_1_t2_unbiased.nii.gz']

