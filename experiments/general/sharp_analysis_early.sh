# /usr/bin/env bash
# This relatively  asimple script just runs an exemplary run to examine loss surface in the beginning of learning

rm $0.batch
mkdir -p $RESULTS_DIR/sharp_analysis

epochsizeresnet=2048
epochsizescnn=512

defaultparams="--m=0 --dropout=0 --fbg_analysis --decompose_analysis --decompose_analysis_kw={'frequency':5} --save_freq=1 --eigen_loss --eigen_loss_ids=[0,2,4] --augmentation --measure_train_loss "
defaultparams="--lr_schedule= --lanczos_top_K=5 --lanczos_top_K_N_sample=2560 --lanczos_aug --lanczos_kwargs={'impl':'scipy'} --n_epochs=2000 --reload $defaultparams"

# Regular
echo "python bin/train_simple_cnn_cifar.py root $RESULTS_DIR/sharp_analysis/scnn_lr=0.01 --lr=0.01 $defaultparams --epoch_size=$epochsizescnn" >> $0.batch
echo "python bin/train_resnet_cifar.py cifar10_resnet32_nobn_nodrop_3 $RESULTS_DIR/sharp_analysis/resnet_lr=0.01 --lr=0.01 $defaultparams --epoch_size=$epochsizeresnet" >> $0.batch
echo "python bin/train_resnet_cifar.py cifar10_resnet32_nobn_nodrop_3 $RESULTS_DIR/sharp_analysis/resnet_lr=0.01_nol2 --l2=0 --lr=0.01 $defaultparams --epoch_size=$epochsizeresnet" >> $0.batch

# 0.05
echo "python bin/train_resnet_cifar.py cifar10_resnet32_nobn_nodrop_3 $RESULTS_DIR/sharp_analysis/resnet_lr=0.05_slow --epoch_size=11250 --lr=0.05 $defaultparams --epoch_size=$epochsizescnn" >> $0.batch
echo "python bin/train_simple_cnn_cifar.py root $RESULTS_DIR/sharp_analysis/scnn_lr=0.05_slow --epoch_size=11250 --lr=0.05 $defaultparams --epoch_size=$epochsizescnn" >> $0.batch

#python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=5