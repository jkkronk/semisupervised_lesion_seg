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
python -u restore_MAP_NN.py --name test_30subj_3e1_10steps_4fch3sh_saug_0 --config conf/conf_nn.yaml --netname 30subj_3e1_10steps_4fch3sh_saug_025 --fprate 0 --subj '['Brats17_TCIA_629_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ANG_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ATX_1_t2_unbiased.nii.gz', 'Brats17_TCIA_321_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ASA_1_t2_unbiased.nii.gz', 'Brats17_TCIA_654_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ASK_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ATV_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ARW_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AOH_1_t2_unbiased.nii.gz', 'Brats17_TCIA_478_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ATP_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ASH_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ASW_1_t2_unbiased.nii.gz', 'Brats17_TCIA_314_1_t2_unbiased.nii.gz', 'Brats17_TCIA_310_1_t2_unbiased.nii.gz', 'Brats17_TCIA_278_1_t2_unbiased.nii.gz']'





