#!/usr/bin/env bash
EXPNAME=mom_baselines

## Huge script running all baselines.

# Configuration
n_epochs=2000
patience=100
espatience=200

l2=0 # Basic version does not use L2
commonparams="--reload --optim=sgd --n_epochs=${n_epochs} --reduce_callback --early_stopping  --lr_schedule=\"\""
commonparams="--l2=${l2} ${commonparams}"

rm $0.batch

sp=$RESULTS_DIR/nsgd/$EXPNAME

mkdir -p $sp

# resnet.sh
for m in 0.9 0.99; do
for config in cifar10_resnet32_nobn_nodrop_3; do
for seed in 777 778; do
for lr in 0.01 0.1; do
    save_path=${sp}/${EXPNAME}_config=${config}_seed=${seed}_overshoot=${overshoot}_lr=${lr}_KK=${KK}_mom=${m}
    mkdir -p $save_path
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_resnet_cifar.py $config $save_path  ${commonparams} --m=${m} --data_seed=${seed} --seed=$seed  --early_stopping_patience=$espatience  --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':$patience}\"  --lr=$lr" >> $0.batch
    fi
done
done
done
done

# scnn.sh
for m in 0.9 0.99; do
for config in medium; do
for seed in 777 778; do
for lr in 0.01 0.1; do
    save_path=${sp}/${EXPNAME}_config=${config}_seed=${seed}_overshoot=${overshoot}_lr=${lr}_KK=${KK}_mom=${m}
    mkdir -p $save_path
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_simple_cnn_cifar.py $config $save_path  ${commonparams} --m=${m} --data_seed=${seed} --seed=$seed  --early_stopping_patience=$espatience  --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':$patience}\"  --lr=$lr" >> $0.batch
    fi
done
done
done
done


python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=3
