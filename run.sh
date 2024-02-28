GPU=${1:-0}
EPOCH=${2:-3}
ROOT=${3:-"./dataset/"}
MODEL=${4:-"BioModel"}
OUTPUT_DIR=${5:-"./output"}
ADDITIONAL=${6:-""}
python run.py \
    --model_type bert \
    --model_name_or_path deepset/deberta-v3-large-squad2 \
    --config_name deepset/deberta-v3-large-squad2 \
    --tokenizer_name deepset/deberta-v3-large-squad2 \
    --load_remote_model \
    --model_class $MODEL\
    --do_train \
    --data_dir $ROOT \
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
    --eval_every_epoch \
    --logging_every_epoch \
    --save_every_epoch \
    --do_eval \
    --predict_file test-set.json \
    --version_2_with_negative \
    --calc_auc \
    --data_augment $ADDITIONAL 
