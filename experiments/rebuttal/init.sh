## 1. Cancel previous jobs

EXPNAME=init2
runsh_path=$0.batch
rm $0.batch

## 2. Create jobs
config=root

sp=$RESULTS_DIR/rebuttal/init/$EXPNAME
mkdir -p $sp

bin="python bin/train_simple_cnn_cifar.py"
m=0
l2=0
K=5
N=2250
lr=0.01
n_epochs=200
bs=128
nepochszoom=4
defaultparams="--reload --optim=sgd --n_epochs=${n_epochs} --m=${m} --lr_schedule=\"\" --batch_size=${bs} --lr=${lr}"
defaultparams="--l2=${l2} ${defaultparams}"
defaultparams="--lanczos_top_K=$K --lanczos_top_K_N_sample=$N --lanczos_kwargs=\"{'impl':'scipy'}\" --lanczos_aug ${defaultparams}"

##### NORMAL #####

for init in glorot_uniform random_uniform random_normal truncated_normal ; do
    save_path=$sp/init=${init}
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
        echo "$bin $config $save_path ${defaultparams} --init=$init" >> $runsh_path
    fi
done

python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=5
