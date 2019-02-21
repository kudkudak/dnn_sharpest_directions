## 1. Cancel previous jobs

## TODO: Fix alingment problem.

EXPNAME=alig_resnet_2
runsh_path=$0.batch
rm $0.batch

## 2. Create jobs
config=cifar10_resnet32_nobn_nodrop_3

sp=$RESULTS_DIR/noise_grid/$EXPNAME
mkdir -p $sp

declare -a mult=(
    "25"
    "1"
    "5"
)
declare -a lrs=(
    "0.1"
    "0.001"
    "0.01"
)
bin="python bin/train_resnet_cifar.py"

m=0
l2=0
K=5
N=2250
epoch_size=45000
lr=0.01
n_epochs=500
bs=128
defaultparams="--reload --optim=sgd --n_epochs=${n_epochs} --m=${m}  --lr_schedule=\"\" --batch_size=${bs} --lr=${lr}"
defaultparams="--l2=${l2} ${defaultparams}"
defaultparams="--fbg_analysis --lanczos_top_K=$K --lanczos_top_K_N_sample=$N --lanczos_kwargs={'impl':'scipy'} --lanczos_aug ${defaultparams}"

for (( i=0; i<${#mult[@]}; i++ )); do
    lr=${lrs[$i]}
    lres=$(($epoch_size / ${mult[$i]}))
    save_path=$sp/constant_lr=${lr}
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        # Hack
        if grep 'EOFError' $save_path/run.sh.out 2> /dev/null; then
            echo "Reset"
            rm $save_path/*
        fi
        if grep 'assert type(e) == type(e_loaded)' $save_path/run.sh.out 2> /dev/null; then
            echo "Reset"
            rm $save_path/*
        fi
        echo "$bin $config $save_path ${defaultparams} --epoch_size=$lres --lr=$lr" >> $runsh_path
    fi
done


