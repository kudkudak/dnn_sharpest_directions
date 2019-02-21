# /usr/bin/env bash
# This relatively  asimple script just runs an exemplary run to examine loss surface in the beginning of learning

rm $0.batch

mkdir -p $RESULTS_DIR/sharp_analysis

epochsizeresnet=2048
epochsizescnn=512

# Doesn't remove L2 in resnet here, just to make sure it doesnt have impace?
defaultparams="--optim=nsgd --opt_kwargs={'overshoot':0.01,'KK':5} --m=0 --dropout=0 --fbg_analysis --decompose_analysis --decompose_analysis_kw={'frequency':5} --save_freq=1 --eigen_loss --eigen_loss_ids=[0,2,4] --augmentation --measure_train_loss "
defaultparams="--lr_schedule= --lanczos_top_K=5 --lanczos_top_K_N_sample=2560 --lanczos_aug --lanczos_kwargs={'impl':'scipy'} --n_epochs=2000 --reload $defaultparams"

echo "python bin/train_simple_cnn_cifar.py root $RESULTS_DIR/sharp_analysis/scnn_lr=0.01_nsgd --lr=0.01 $defaultparams --epoch_size=$epochsizescnn" >> $0.batch
echo "python bin/train_resnet_cifar.py cifar10_resnet32_nobn_nodrop_3 $RESULTS_DIR/sharp_analysis/resnet_lr=0.01_nsgd --lr=0.01 $defaultparams --epoch_size=$epochsizeresnet" >> $0.batch
