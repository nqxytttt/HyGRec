#!/bin/bash

DATASET=Instruments
BASE_MODEL=
DATA_PATH=../data
OUTPUT_DIR=./$DATASET/

torchrun --nproc_per_node=8 --master_port=13325  lora_finetune.py \
    --base_model $BASE_MODEL\
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --per_device_batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 5 \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .index.json\
    --wandb_run_name test\
    --temperature 1.0
