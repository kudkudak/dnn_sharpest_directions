#!/usr/bin/env bash

# Main design principle: we do actually reach a sharper/wider region. And the initial difference in sharpness is large
# Only then we have some chance it matters

EXPNAME=csgd_imdb7

rm $0.batch
sp=$RESULTS_DIR/rebuttal/$EXPNAME
mkdir -p $sp

# Configuration
n_epochs=1000
K=5
N=2560
patience=100
espatience=50
bs=128
m=0
commonparams="--reload --optim=nsgd --n_epochs=${n_epochs} --reduce_callback --early_stopping  --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':$patience}\" --early_stopping_patience=$espatience --m=${m} --lr_schedule=\"\""
commonparams="--lanczos_top_K=$K --lanczos_top_K_N_sample=$N --no_lanczos_inference_mode --lanczos_kwargs=\"{'impl':'scipy'}\" ${commonparams}"


setparams=""

for config in cnn_big; do
for dropout in 0.0; do
for seed in 778 779 780; do
for overshoot in 0.1 1.0 5; do
for KK in 1; do
for lr in 0.01; do
for bs in 8; do
    save_path=${sp}/${EXPNAME}_config=${config}_seed=${seed}_overshoot=${overshoot}_lr=${lr}_KK=${KK}_dropout=${dropout}_bs=$bs
    mkdir -p $save_path
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_imdb.py $config $save_path  ${commonparams} --batch_size=$bs --dropout=$dropout --data_seed=${seed} --seed=$seed  --lr=$lr --opt_kwargs=\"{'overshoot':${overshoot},'KK':$KK}\"" >> $0.batch
    fi
done
done
done
done
done
done
done

python bin/utils/run_slurm.py $sp --use_qos_unkillable=1 --use_qos_high=2 --list_path=$0.batch --name=$EXPNAME --max_jobs=12
