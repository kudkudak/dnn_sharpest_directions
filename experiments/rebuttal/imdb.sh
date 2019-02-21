#!/usr/bin/env bash
EXPNAME=imdb2

rm $0.batch
sp=$RESULTS_DIR/rebuttal/$EXPNAME
mkdir -p $sp

# Configuration
n_epochs=1000
K=2
N=2560
patience=100
espatience=100
bs=128
m=0
commonparams="--reload --optim=sgd --n_epochs=${n_epochs} --reduce_callback --early_stopping  --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':$patience}\" --early_stopping_patience=$espatience --m=${m} --lr_schedule=\"\""
commonparams="--lanczos_top_K=$K --lanczos_top_K_N_sample=$N --no_lanczos_inference_mode --lanczos_kwargs=\"{'impl':'scipy'}\" ${commonparams}"

setparams=""

for config in cnn; do
for dropout in 0.0; do
for seed in 777; do
for lr in 0.01; do
for bs in 256 32 8 2; do
    save_path=${sp}/${EXPNAME}_config=${config}_seed=${seed}_lr=${lr}_dropout=${dropout}_bs=$bs
    mkdir -p $save_path
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_imdb.py $config $save_path  ${commonparams} --batch_size=$bs --dropout=$dropout --data_seed=${seed} --seed=$seed  --lr=$lr" >> $0.batch
    fi
done
done
for lr in 0.1 0.05 0.025 0.01; do
for bs in 32; do
    save_path=${sp}/${EXPNAME}_config=${config}_seed=${seed}_lr=${lr}_KK=${KK}_dropout=${dropout}_bs=$bs
    mkdir -p $save_path
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_imdb.py $config $save_path  ${commonparams} --batch_size=$bs --dropout=$dropout --data_seed=${seed} --seed=$seed  --lr=$lr" >> $0.batch
    fi
done
done

done
done
done

python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=8
