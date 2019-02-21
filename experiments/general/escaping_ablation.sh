#!/usr/bin/env bash

E=$RESULTS_DIR/simple_cnn/eigenloss_batch_lr=0.005_nomom
epoch=395
N=1000

mkdir $RESULTS_DIR/simple_cnn/escaping_ablation

python experiments/simple_cnn/escaping_ablation.py $ISOTROPY_RESULTS_DIR/simple_cnn/escaping_ablation2/normal_n=$N_track_deltaW --track_deltaW --which=normal --E=$E --epoch=$epoch --N=$N --id=0
python experiments/simple_cnn/escaping_ablation.py $ISOTROPY_RESULTS_DIR/simple_cnn/escaping_ablation2/normal_n=$N --which=normal --E=$E --epoch=$epoch --N=$N --id=0
python experiments/simple_cnn/escaping_ablation.py $ISOTROPY_RESULTS_DIR/simple_cnn/escaping_ablation2/onlytop_n=$N --which=onlytop --E=$E --epoch=$epoch --N=$N --id=0
python experiments/simple_cnn/escaping_ablation.py $ISOTROPY_RESULTS_DIR/simple_cnn/escaping_ablation2/constanttop_n=$N --which=constanttop --E=$E --epoch=$epoch --N=$N --id=0
python experiments/simple_cnn/escaping_ablation.py $ISOTROPY_RESULTS_DIR/simple_cnn/escaping_ablation2/removetop_n=$N --which=removetop --E=$E --epoch=$epoch --N=$N --id=0
