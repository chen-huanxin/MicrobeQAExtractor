GPU=${1:-0}
EPOCH=${2:-3}
DATA_DIR=${3:-"./dataset/"}
MODEL=${4:-"BioModel"}
OUTPUT_DIR=${5:-"./output"}

python run.py \
    --model_type bert \
    --model_name_or_path models/deberta-v3-base-microbedb-v1 \
    --load_remote_model \
    --model_class $MODEL\
    --do_train \
    --data_dir $DATA_DIR \
    --train_file train-set.json \
    --per_gpu_train_batch_size 4 \
    --learning_rate 8e-6 \
    --num_train_epochs $EPOCH \
    --max_seq_length 384 \
    --seed 0 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --single_gpu \
    --gpu $GPU \
    --logging_every_epoch \
    --save_every_epoch \
    --do_eval \
    --predict_file test-set.json \
    --data_augment
