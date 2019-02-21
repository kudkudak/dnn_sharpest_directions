#!/usr/bin/env bash
EXPNAME=highlr_resnet

## 15h na run?

rm $0.batch
sp=$RESULTS_DIR/nsgd/$EXPNAME
mkdir -p $sp

# Configuration
n_epochs=2000
K=10 # To be safe
N=2560
patience=100
espatience=200
m=0
l2=0
commonparams="--reload --optim=nsgd --n_epochs=${n_epochs} --reduce_callback --early_stopping  --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':$patience}\" --early_stopping_patience=$espatience --m=${m} --lr_schedule=\"\""
commonparams="--l2=${l2} ${commonparams}"
commonparams="--lanczos_top_K=$K --lanczos_top_K_N_sample=$N --lanczos_kwargs=\"{'impl':'scipy'}\" --lanczos_aug ${commonparams}"

setparams=""

# Run no mom, no L2, CIFAR10
for config in cifar10_resnet32_nobn_nodrop_3; do # Later cifar10_resnet56_nobn_nodrop_3 ?
for seed in 777 778; do
for overshoot in 0.01 0.1 1.0 5; do
for KK in 5; do
for lr in 0.1; do
    save_path=${sp}/${EXPNAME}_config=${config}_seed=${seed}_overshoot=${overshoot}_lr=${lr}_KK=${KK}
    mkdir -p $save_path
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_resnet_cifar.py $config $save_path  ${commonparams} --data_seed=${seed} --seed=$seed  --lr=$lr --opt_kwargs=\"{'overshoot':${overshoot},'KK':$KK}\"" >> $0.batch
    fi
done
done
done
done
done

python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=4
