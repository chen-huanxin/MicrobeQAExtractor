# MicrobeQAExtractor

To train QA models on MicrobeDB with BioBERT-v1.1 or DeBERTa-v3, follow the description below.

## Models
- `ktrapeznikov/biobert_v1.1_pubmed_squad_v2`
- `deepset/deberta-v3-base-squad2`

## Configuration

### Configure Environment
```bash
conda env create -f requirements.yaml
conda activate qa
```
### Additional Requirements
- Transforms
- pandas : Transforms the SQuAD prediction file into the BioASQ format (`pip install pandas`)
- tensorboardX : SummaryWriter module (`pip install tensorboardX`)

### Dataset
Default dataset directory: `./dataset`. 
Place `train-set.json` and `test-set.json` in dataset directory.


*Tips: You can change the file names by `--train_file`, `--predict_file`*

### Download Pre-trained model
- Download the pre-trained model [pytorch_model.bin](https://huggingface.co/ktrapeznikov/biobert_v1.1_pubmed_squad_v2/resolve/main/pytorch_model.bin)
- Place `pytorch_model.bin` in project's `models/` directory

### Run Command
`--data_dir` defines which directory the dataset is in
```bash
python run.py \
    --model_type bert \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --model_class BioModelQkv \  # choices: [BioModel, BioModelQkv, BioModelQkvBiDirection, BioModelQkvBiDirectionResidual]
    --data_dir ./dataset \
    --per_gpu_train_batch_size 12 \
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
    --train_file train-set.json \

# Evaluation
    --do_eval
    --predict_file test-set.json
```
Or just run the preject by shell script `run.sh`
```bash
./run.sh [GPU] [EPOCH] [ROOT] [MODEL]
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