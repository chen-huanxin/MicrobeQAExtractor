# MicrobeQAExtractor

This repository provides the code for pathogenic microorganism knowledge extraction and the trained model weights, along with several illustrative test samples.

## Models
- [`deberta-v3-base-microbedb-v1`](https://drive.google.com/drive/folders/1t8Q6P_6WsSn6XRP9TZXBzaU5w_mgN0TK?usp=drive_link)
- [`biobert_v1.1_microbedb_v1`](https://drive.google.com/drive/folders/1ZMQ90Bx1cNxQbIKaGyCrRZW9HSWWBMoM?usp=drive_link)

## Configuration

### Configure Environment
```bash
conda env create -f requirements.yaml
conda activate qa
```

<!-- ### Additional Requirements
- Transforms
- pandas : Transforms the SQuAD prediction file into the BioASQ format (`pip install pandas`)
- tensorboardX : SummaryWriter module (`pip install tensorboardX`) -->

### Dataset
Default dataset directory: `./dataset`. 
Place `train-set.json` and `test-set.json` in dataset directory.

*Tips: You can change the file names by `--train_file`, `--predict_file`*

### Download Pre-trained model
- Download the pre-trained model
- Place model in project's `models/` directory

### Run Command
`--data_dir` defines which directory the dataset is in
```bash
python run.py \
    --model_type bert \
    --model_name_or_path models/deberta-v3-base-microbedb-v1 \
    --load_remote_model \
    --model_class BioModel \  
    --data_dir ./dataset \
    --learning_rate 8e-6 \
    --num_train_epochs 3 \
    --max_seq_length 384 \
    --seed 0 \
    --output_dir ./output
    --overwrite_output_dir
    --single_gpu \
    --gpu 0 \
# Train
    --do_train \
    --per_gpu_train_batch_size 12 \
    --train_file train-set.json \

# Evaluation
    --do_eval
    --predict_file test-set.json
```
Or just run the preject by shell script `run.sh`
```bash
./run.sh [GPU] [EPOCH] [DATA_DIR] [MODEL]
# like
./run.sh                # default: Use No.0 GPU, run 3 epoches, the dataset is in the directory ./dataset
./run.sh 1              # Use No.1 GPU
./run.sh 2 10           # Use No.2 GPU, run 10 epoches
./run.sh 3 20 ../dataset # Use No.3 GPU, run 20 epoches, the dataset is in the directory ../dataset
```

### Related work
This code comes from related work: **Interpretation knowledge extraction for genetic testing via question-answer model**

Authors: Wenjun Wang†, Huanxin Chen†, Hui Wang†, Lin Fang, Huan Wang, Yi Ding*, Yao Lu* and Qingyao Wu*

†: Equal contributor

*: Correspondent author

### Contact
For help or issues using MicrobeQAExtractor, please create an issue.