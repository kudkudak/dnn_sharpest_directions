#!/usr/bin/env bash
EXPNAME=scnn_highlr

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
l2=0 # Basic version does not use L2
commonparams="--reload --optim=nsgd --n_epochs=${n_epochs} --reduce_callback --early_stopping --early_stopping_patience=$espatience --m=${m} --lr_schedule=\"\""
commonparams="--l2=${l2} ${commonparams}"
commonparams="--lanczos_top_K=$K --lanczos_top_K_N_sample=$N --lanczos_kwargs=\"{'impl':'scipy'}\" --lanczos_aug ${commonparams}"

# Run no mom, no L2
id=0
for config in medium; do
for seed in 777 778; do
for overshoot in 0.01 0.1 1.0 5; do
for KK in 10; do #1
for lr in 0.1; do # 0.1
    save_path=${sp}/${EXPNAME}_config=${config}_seed=${seed}_overshoot=${overshoot}_lr=${lr}_KK=${KK}
    mkdir -p $save_path
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_simple_cnn_cifar.py $config $save_path  ${commonparams} --data_seed=${seed}  --seed=$seed    --reduce_callback_kwargs=\"{'monitor': 'val_loss', 'patience':$patience}\"  --lr=$lr --opt_kwargs=\"{'overshoot':${overshoot},'KK':$KK}\"" >> $0.batch
    fi
    id=$((${id} + 1))
    tail -n 1 $0.batch > $0.batch.${id}
done
done
done
done
done

python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=8
