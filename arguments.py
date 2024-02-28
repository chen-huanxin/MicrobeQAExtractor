import os
import argparse

from transformers import MODEL_FOR_QUESTION_ANSWERING_MAPPING

from models.derived_models import (
    BioModel, BioModelQkv, 
    BioModelQkvBiDirection, 
    BioModelQkvBiDirectionResidual, 
    BioModelClassifyOne, 
    BioModelClassifyTwo, 
    BioModelClassifyCNN,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
MODEL_CLASS_TABLE = {
    "BioModel": BioModel, 
    "BioModelQkv": BioModelQkv, 
    "BioModelQkvBiDirection": BioModelQkvBiDirection,
    "BioModelQkvBiDirectionResidual": BioModelQkvBiDirectionResidual,
    "BioModelClassifyOne": BioModelClassifyOne,
    "BioModelClassifyTwo": BioModelClassifyTwo,
    "BioModelClassifyCNN": BioModelClassifyCNN,
}

def get_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument( # bert
        "--model_type",
        default="bert",
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument( # dmis-lab/biobert-base-cased-v1.1
        "--model_name_or_path",
        default="ktrapeznikov/biobert_v1.1_pubmed_squad_v2",
        type=str,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument( # output
        "--output_dir",
        default=os.path.join(os.getcwd(), "output"),
        type=str,
        # required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="BioModel",
        choices=MODEL_CLASS_TABLE.keys(),
        help="The model class to use",
    )
    parser.add_argument( # 用于验证，还没定义
        "--golden_file",
        default=None,
        type=str,
        help="BioASQ official golden answer file"
    )
    parser.add_argument( # 用于验证，还没定义
        "--official_eval_dir",
        default='./scripts/bioasq_eval',
        type=str,
        help="BioASQ official golden answer file"
    )
    
    # Other parameters
    parser.add_argument( # ../datasets/QA/BioASQ/
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument( # BioASQ-train-yesno-7b.json / train-set.json
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument( # test-set.json
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument( # 没定义，直接用model_name_or_path的值
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument( # 没定义，直接用model_name_or_path的值
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument( # ../data-cache
        "--cache_dir",
        default="data-cache",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument( # 第二版，包含is_impossible的数据集
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.", # 包含没答案的样本
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.", # 如果null_score的值比阈值高，就预测为空
    )

    parser.add_argument( # 384
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument( # 没定义，用默认值
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument( 
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.") 
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--eval_every_x_step", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.") # 12
    parser.add_argument( 
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.") # 8e-6
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=0, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    # parser.add_argument("--eval_epoches", type=int, default=10, help="Evaluate model every X epoches. ")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument( # 是否要覆盖output目录的内容
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization") # 0

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus") # 用于分布式训练的，这里用不上，-1表示不使用
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", # 是否使用半精度浮点数进行训练
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument("--single_gpu", action="store_true", help="Just use one gpu")
    parser.add_argument("--gpu", type=int, default=0, help="Which gpu to use")
    parser.add_argument("--use_exist_model", action="store_true", help="Use exist model to train")
    parser.add_argument("--exist_model_path", type=str, default="pytorch_model_qkv.bin", help="Exist model path")
    parser.add_argument("--eval_every_epoch", action="store_true", default=False, help="Evaluate every epoch")
    parser.add_argument("--logging_every_epoch", action="store_true", default=True, help="Show log every epoch")
    parser.add_argument("--save_every_epoch", action="store_true", default=False, help="Save model every epoch")
    parser.add_argument("--data_augment", action="store_true", default=False, help="augment train-set")
    parser.add_argument("--calc_auc", action="store_true", default=False, help="calcuate AUC")
    parser.add_argument("--threshold_start", type=float, default=-40.0, help="threshold start value")
    parser.add_argument("--threshold_end", type=float, default=40.0, help="threshold end value")
    parser.add_argument("--threshold_step", type=float, default=8.0, help="threshold step value")
    parser.add_argument("--load_remote_model", action="store_true", default=False, help="load model from remote")
    parser.add_argument("--use_distloss", action="store_true", default=False, help="Whether use distloss func(with position_nums params when training)") # 添加这一个选项，输入模型的参数会多`position_nums`这一项

    return  parser.parse_args()