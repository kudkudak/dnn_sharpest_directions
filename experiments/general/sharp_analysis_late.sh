# /usr/bin/env bash

rm $0.batch

mkdir -p $RESULTS_DIR/sharp_analysis

# Doesn't remove L2 in resnet here, just to make sure it doesnt have impace?
defaultparams="--m=0 --dropout=0 --fbg_analysis --decompose_analysis --decompose_analysis_kw={'frequency':5} --save_freq=1 --eigen_loss --eigen_loss_ids=[0,2,4] --augmentation --measure_train_loss --epoch_size=-1"
defaultparams="--lr_schedule= --lanczos_top_K=5 --lanczos_top_K_N_sample=2560 --lanczos_aug --lanczos_kwargs={'impl':'scipy'} --n_epochs=300 --reload $defaultparams"

# Create jobs
echo "python bin/train_simple_cnn_cifar.py root $RESULTS_DIR/sharp_analysis/scnn_lr=0.01_long --lr=0.01 $defaultparams " >> $0.batch
echo "python bin/train_simple_cnn_cifar.py root $RESULTS_DIR/sharp_analysis/scnn_lr=0.05_long --lr=0.05 $defaultparams" >> $0.batch
echo "python bin/train_resnet_cifar.py cifar10_resnet32_nobn_nodrop_3 $RESULTS_DIR/sharp_analysis/resnet_lr=0.01_logg --lr=0.01 $defaultparams" >> $0.batch
echo "python bin/train_resnet_cifar.py cifar10_resnet32_nobn_nodrop_3 $RESULTS_DIR/sharp_analysis/resnet_lr=0.05_long  --lr=0.05 $defaultparams" >> $0.batch
#python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=5