## 1. Cancel previous jobs

EXPNAME=ng_ptb3
runsh_path=$0.batch
rm $0.batch

## 2. Create jobs
config=small

sp=$RESULTS_DIR/noise_grid/$EXPNAME
mkdir -p $sp

m=0
K=5 # PTB takes a lot of time, but not that long.
N=500 # PTB is larger, 1% is enough
lrs=""
bss="64 256"
keep_prob=0.9 # Basically for avoiding 0-out loss
n_epochs=400
bs=20
epochsize=46000 # Roughly, assuming nsteps=20. PTB has ~900k words, so 900k/20 ~= 46k. (Orders of magnitude)
smallepochsize=256
espatience=50
nepochszoom=4
defaultparams="--keep_prob=${keep_prob} --early_stopping --early_stopping_patience=$espatience --reload --opt=sgd --n_epochs=${n_epochs} --m=${m} --lr_schedule= --batch_size=${bs}"
defaultparams="--lanczos_top_K=$K --lanczos_top_K_N_sample=$N --lanczos_kwargs={'impl':'scipy'} --lanczos_aug ${defaultparams}"

##### NORMAL #####

for lr in $lrs; do
    save_path=$sp/constant_lr=${lr}
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_ptb_lm.py $config $save_path ${defaultparams} --lr=$lr" >> $runsh_path
    fi
done

lr=1.0
for bs in $bss; do
    save_path=$sp/constant_bs=${bs}_lr=${lr}
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_ptb_lm.py $config $save_path ${defaultparams} --lr=${lr} --lanczos_top_K_bs=$bs --batch_size=${bs}" >> $runsh_path
    fi
done

##############

#### ZOOM ####

smallepochsize=$(($epochsize/100))
n_epochs=$((${nepochszoom}*45000/${smallepochsize} + 1)) # 10 epochs

echo "Will run for ${n_epochs}"

for lr in $lrs; do
    save_path=$sp/zoom_constant_lr=${lr}
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_ptb_lm.py $config $save_path ${defaultparams} --lr=$lr  --measure_train_loss --epoch_size=${smallepochsize} --n_epochs=${n_epochs} --reload" >> $runsh_path
    fi
done

lr=1.0
for bs in $bss; do
    n_epochs=$((${nepochszoom}*$epochsize/$bs + 1)) # 10 epochs
    save_path=$sp/zoom_constant_bs=${bs}_lr=${lr}
    if grep 'Finished train' $save_path/stderr.txt 2> /dev/null; then
        echo "Finished $save_path"
    else
        echo "python bin/train_ptb_lm.py $config $save_path ${defaultparams} --lr=$lr  --measure_train_loss --epoch_size=${smallepochsize} --lanczos_top_K_bs=$bs --batch_size=${bs} --n_epochs=${n_epochs}" >> $runsh_path
    fi
done

#python bin/utils/run_slurm.py $sp --list_path=$0.batch --name=$EXPNAME --max_jobs=5
