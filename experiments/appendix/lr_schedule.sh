#!/usr/bin/env bash
EXPNAME=lrsch_resnet

rm $0.batch
sp=$RESULTS_DIR/nsgd/$EXPNAME
mkdir -p $sp

# Configuration
n_epochs=200
K=5 # To be safe
N=2560
patience=100
espatience=200
m=0
l2=0
commonparams="--reload --n_epochs=${n_epochs}  --m=${m} "
commonparams="--l2=${l2} ${commonparams}"
commonparams="--lanczos_top_K=$K --lanczos_top_K_N_sample=$N --lanczos_kwargs=\"{'impl':'scipy'}\" --lanczos_aug ${commonparams}"

setparams=""

for config in cifar10_resnet32_nobn_nodrop_3; do # Later cifar10_resnet56_nobn_nodrop_3 ?
for seed in 777; do
for L in 20 40 80 160; do
for lr in 0.1; do
    save_path=${sp}/${EXPNAME}_config=${config}_L=${L}_seed=${seed}_overshoot=${overshoot}_lr=${lr}_KK=${KK}
    mkdir -p $save_path
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_resnet_cifar.py $config $save_path  ${commonparams} --data_seed=${seed} --seed=$seed  --lr_schedule=\"[[${L}, 0.1],[${L}+40, 0.01]]\"" >> $0.batch
    fi
done
done
done
done

python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=4
