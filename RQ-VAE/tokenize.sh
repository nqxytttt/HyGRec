#!/bin/bash
python ./generate_indices.py\
    --dataset Instruments \
    --alpha 1e-1 \
    --beta 1e-4 \
    --epoch 10000 \
    --checkpoint best_collision_model.pth