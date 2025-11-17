#!/bin/bash

python train.py \
    data/large_gensfen_multipvdiff_100_d9.binpack \
    --gpus 0 \
    --max_epochs 1 \
    --batch-size 16 \
    --epoch-size 100 \
    --validation-size 20 \
    --num-workers 1 \
    --threads 1 \
    --seed 42 \
    --no-smart-fen-skipping \
    --no-wld-fen-skipping \
    --random-fen-skipping 0 \
    --features "HalfKAv2_hm^" \
    --lambda 1.0 \
    --lr 0.001 \
    --default_root_dir "./logs_small_test"