#!/bin/bash
nqueries_list=(16 32 64 128 196 256 384 512)  # Define your nqueries values here

for nqueries in "${nqueries_list[@]}"; do
    echo "Running test with nqueries = $nqueries"
    python main.py --dataset_name sunrgbd \
        --dataset_root_dir sunrgbd_pc_bbox_votes_50k_v2 \
        --nqueries $nqueries \
        --test_ckpt ./checkpoints/sunrgbd_masked_ep1080.pth \
        --test_only \
        --enc_type masked \
        --batchsize_per_gpu=32
done
