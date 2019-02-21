## 1. Cancel previous jobs

### Note: changes bs

EXPNAME=ng_vgg11
runsh_path=$0.batch
rm $0.batch

## 2. Create jobs
config=vgg11_no_dropout

sp=$RESULTS_DIR/noise_grid/$EXPNAME
mkdir -p $sp

m=0
l2=0
K=5
N=2250
lr=0.01
n_epochs=200
bs=128
lrs="0.1 0.01 0.001"
bss="8 128 512"
nepochszoom=4
defaultparams="--reload --optim=sgd --n_epochs=${n_epochs} --m=${m} --lr_schedule=\"\" --batch_size=${bs} --lr=${lr}"
defaultparams="--l2=${l2} ${defaultparams}"
defaultparams="--lanczos_top_K=$K --lanczos_top_K_N_sample=$N --lanczos_kwargs=\"{'impl':'scipy'}\" --lanczos_aug ${defaultparams}"

##### NORMAL #####

for lr in $lrs; do
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
        echo "python bin/train_vgg_cifar.py $config $save_path ${defaultparams} --lr=$lr" >> $runsh_path
    fi
done

lr=0.05
for bs in $bss; do
    save_path=$sp/constant_bs=${bs}_lr=${lr}
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

        echo "python bin/train_vgg_cifar.py $config $save_path ${defaultparams} --lr=${lr} --batch_size=${bs}" >> $runsh_path
    fi
done

##############

#### ZOOM ####

epoch_size=512 # 4 batches
n_epochs=$((${nepochszoom}*45000/${epoch_size} + 1)) # 10 epochs

echo "Will run for ${n_epochs}"

for lr in $lrs; do
    save_path=$sp/zoom_constant_lr=${lr}
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

        echo "python bin/train_vgg_cifar.py $config $save_path ${defaultparams} --lr=$lr  --measure_train_loss --epoch_size=${epoch_size} --n_epochs=${n_epochs} --reload" >> $runsh_path
    fi
done

lr=0.05
epoch_size=512
for bs in $bss; do
    n_epochs=$((${nepochszoom}*45000/$bs + 1)) # 10 epochs
    save_path=$sp/zoom_constant_bs=${bs}_lr=${lr}
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

        echo "python bin/train_vgg_cifar.py $config $save_path ${defaultparams} --lr=$lr  --measure_train_loss --epoch_size=${epoch_size} --batch_size=${bs} --n_epochs=${n_epochs}" >> $runsh_path
    fi
done

#python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=5
