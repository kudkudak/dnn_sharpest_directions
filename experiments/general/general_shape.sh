# /usr/bin/env bash
# Subset of noise grid specific for Sec.3

EXPNAME=general_shape

rm $0.batch

### General config

sd=$RESULTS_DIR/general_shape
sp=sd

# Keeping L2

lrsimple=0.1
lrsimple2=0.01
lrresnet=0.01
lrresnet2=0.1
sf=100 # Actually save!
m=0 # Momentum
n_epochs=400
n_epochs_zoom=5
K=10
N=2560 # Relatively large to have a stable estimation

defaultparams="--save_freq=$sf --lr_schedule="" --m=$m --dropout=0 --n_epochs=${n_epochs} --reload"
defaultparams="--lanczos_top_K=$K --lanczos_aug  --lanczos_kwargs=\"{'impl':'scipy'}\" --lanczos_top_K_N_sample=$N  --lanczos_top_K_bs=128 $defaultparams"

## 2. Create jobs

config=cifar10_resnet32_nobn_nodrop_3

##### Resnet-32 #####

for lr in $lrresnet2; do
    save_path=$sd/resnet32_constant_l2_lr=${lr}
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
        echo "python bin/train_resnet_cifar.py $config $save_path $defaultparams --lr=$lr" >> $0.batch
    fi
done

epoch_size=128 # Very frequent
n_epochs=$((${n_epochs_zoom}*45000/${epoch_size} + 1)) # 3 epochs

echo "Will run for ${n_epochs}"

for lr in $lrresnet2; do
    save_path=$sd/resnet32_zoom_constant_l2_lr=${lr}
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        if grep 'EOFError' $save_path/run.sh.out 2> /dev/null; then
            echo "Reset"
            rm $save_path/*
        fi
        if grep 'assert type(e) == type(e_loaded)' $save_path/run.sh.out 2> /dev/null; then
            echo "Reset"
            rm $save_path/*
        fi
        echo "python bin/train_resnet_cifar.py $config $save_path  $defaultparams   --measure_train_loss --epoch_size=${epoch_size} --lr=$lr --n_epochs=${n_epochs}" >> $0.batch
    fi
done


### SimpleCNN ###

config=root
epoch_size=-1
n_epochs=300 # Not sure why

for lr in $lrsimple2; do # 0.1 # One very small
    save_path=$sd/simplecnn_constant_lr=${lr}
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        if grep 'EOFError' $save_path/run.sh.out 2> /dev/null; then
            echo "Reset"
            rm $save_path/*
        fi
        if grep 'assert type(e) == type(e_loaded)' $save_path/run.sh.out 2> /dev/null; then
            echo "Reset"
            rm $save_path/*
        fi

        if grep 'assert len(model.history.history) != 0' $save_path/run.sh.out 2> /dev/null; then
        echo "Reset"
            rm $save_path/*
        fi
        echo "python bin/train_simple_cnn_cifar.py $config $save_path $defaultparams --lr=$lr --n_epochs=${n_epochs}" >> $0.batch
    fi
done

epoch_size=128 # Very frequent
n_epochs=$((${n_epochs_zoom}*45000/${epoch_size} + 1)) # 3 epochs

for lr in $lrsimple2; do
    save_path=$sd/simplcnn_zoom_constant_lr=${lr}
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        if grep 'EOFError' $save_path/run.sh.out 2> /dev/null; then
            echo "Reset"
            rm $save_path/*
        fi
        if grep 'assert type(e) == type(e_loaded)' $save_path/run.sh.out 2> /dev/null; then
            echo "Reset"
            rm $save_path/*
        fi
        if grep 'assert len(model.history.history) != 0' $save_path/run.sh.out 2> /dev/null; then
        echo "Reset"
            rm $save_path/*
        fi
        echo "python bin/train_simple_cnn_cifar.py $config $save_path $defaultparams --measure_train_loss --lr=$lr --epoch_size=${epoch_size} --n_epochs=${n_epochs}" >> $0.batch
    fi
done

python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=3


