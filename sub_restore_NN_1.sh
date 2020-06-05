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
python -u restore_MAP_NN.py --name test_10subj_1K_4lr_15steps_norm_healthy_ --config conf/conf_nn.yaml --netname 10subj_1K_4lr_15steps_norm40 --fprate 0.05 --subj '['Brats17_TCIA_276_1_t2_unbiased.nii.gz', 'Brats17_TCIA_396_1_t2_unbiased.nii.gz', 'Brats17_CBICA_BFP_1_t2_unbiased.nii.gz', 'Brats17_TCIA_378_1_t2_unbiased.nii.gz', 'Brats17_TCIA_221_1_t2_unbiased.nii.gz', 'Brats17_TCIA_343_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQZ_1_t2_unbiased.nii.gz', 'Brats17_TCIA_462_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ABY_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AVG_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_10subj_1K_4lr_15steps_norm_comp_ --config conf/conf_nn.yaml --netname 10subj_1K_4lr_15steps_norm40 --fprate 0 --subj '['Brats17_TCIA_276_1_t2_unbiased.nii.gz', 'Brats17_TCIA_396_1_t2_unbiased.nii.gz', 'Brats17_CBICA_BFP_1_t2_unbiased.nii.gz', 'Brats17_TCIA_378_1_t2_unbiased.nii.gz', 'Brats17_TCIA_221_1_t2_unbiased.nii.gz', 'Brats17_TCIA_343_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQZ_1_t2_unbiased.nii.gz', 'Brats17_TCIA_462_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ABY_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AVG_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_25subj_1K_4lr_15steps_norm_comp_ --config conf/conf_nn.yaml --netname 25subj_1K_4lr_15steps_norm10 --fprate 0 --subj '['Brats17_TCIA_499_1_t2_unbiased.nii.gz', 'Brats17_TCIA_192_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ANG_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQQ_1_t2_unbiased.nii.gz', 'Brats17_TCIA_466_1_t2_unbiased.nii.gz', 'Brats17_2013_14_1_t2_unbiased.nii.gz', 'Brats17_TCIA_624_1_t2_unbiased.nii.gz', 'Brats17_CBICA_BFB_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ATB_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ARW_1_t2_unbiased.nii.gz']'

python -u restore_MAP_NN.py --name test_25subj_1K_4lr_15steps_norm_healthy_ --config conf/conf_nn.yaml --netname 25subj_1K_4lr_15steps_norm10 --fprate 0.05 --subj '['Brats17_TCIA_499_1_t2_unbiased.nii.gz', 'Brats17_TCIA_192_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ANG_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQQ_1_t2_unbiased.nii.gz', 'Brats17_TCIA_466_1_t2_unbiased.nii.gz', 'Brats17_2013_14_1_t2_unbiased.nii.gz', 'Brats17_TCIA_624_1_t2_unbiased.nii.gz', 'Brats17_CBICA_BFB_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ATB_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ARW_1_t2_unbiased.nii.gz', 'Brats17_TCIA_412_1_t2_unbiased.nii.gz', 'Brats17_TCIA_436_1_t2_unbiased.nii.gz', 'Brats17_TCIA_290_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ABB_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQA_1_t2_unbiased.nii.gz', 'Brats17_TCIA_211_1_t2_unbiased.nii.gz', 'Brats17_TCIA_325_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ASG_1_t2_unbiased.nii.gz', 'Brats17_TCIA_632_1_t2_unbiased.nii.gz', 'Brats17_CBICA_ATX_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQY_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AQJ_1_t2_unbiased.nii.gz', 'Brats17_TCIA_280_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AOZ_1_t2_unbiased.nii.gz', 'Brats17_CBICA_AXO_1_t2_unbiased.nii.gz']'



