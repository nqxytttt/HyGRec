#!/bin/bash

DATASET=Yelp
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/ddp.json
BASE_MODEL



torchrun --nproc_per_node=8 --master_port=4324  test_ddp.py \
    --ckpt_path   \
    --base_model $BASE_MODEL\
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 40 \
    --test_prompt_ids 0 \
    --index_file .index.json