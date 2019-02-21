#!/usr/bin/env bash

E=$RESULTS_DIR/simple_cnn/eigenloss_batch_lr=0.005_nomom
epoch=395
N=1000

mkdir $RESULTS_DIR/simple_cnn/escaping_ablation_topbottom

for factor in 4 2 0.5 0.25; do
    python experiments/simple_cnn/escaping_ablation.py $ISOTROPY_RESULTS_DIR/simple_cnn/escaping_ablation_topbottom/top_normal_n=${N}_factor=${factor} --lrfactor=$factor --which=topscaled --E=$E --epoch=$epoch --N=$N --id=0
done

for factor in 4 2 0.5 0.25; do
    python experiments/simple_cnn/escaping_ablation.py $ISOTROPY_RESULTS_DIR/simple_cnn/escaping_ablation_topbottom/bottom_normal_n=${N}_factor=${factor} --lrfactor=$factor --which=bottomscaled --E=$E --epoch=$epoch --N=$N --id=0
done
