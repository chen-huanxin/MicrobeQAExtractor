import os
import torch
import logging

from transformers.data.processors.squad import SquadV1Processor

from processors.preprocess import BioProcessor, squad_convert_examples_to_features

# from processors.distloss_preproc import BioProcessor, squad_convert_examples_to_features
from processors.distloss_preproc_plus import BioProcessor as BioProcessorDistloss, squad_convert_examples_to_features as squad_convert_examples_to_features_distloss


logger = logging.getLogger("BIOBERT")


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):

    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."  
    name = os.path.basename(args.predict_file if evaluate else args.train_file)
    stem, _ = os.path.splitext(name) # 获取文件名

    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}{}{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            stem,
            str(args.max_seq_length),
            "_augment" if not evaluate and args.data_augment else "",
            "_distloss" if args.use_distloss else "",
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache: # 如果之前已经生成了cache，没有清除，不会进入该分支
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)): # 用不到，不会进入，因为我们定义了args.train_file
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            if args.use_distloss:
                processor = BioProcessorDistloss()
            else:
                processor = BioProcessor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
                check_example(examples)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file, augment=args.data_augment)
                check_example(examples)

        if args.use_distloss:
            features, dataset = squad_convert_examples_to_features_distloss(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
                return_dataset="pt",
                threads=args.threads,
            )
        else:
            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
                return_dataset="pt",
                threads=args.threads,
            )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    logger.info("** Load Data Success *******************************")
    logger.info("** Example Info ************************************")
    logger.info(f"   Samples num: {len(examples)}")

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

def check_example(examples: list):
    logger.info(f"Now example len: {len(examples)}")

    is_impossible_num = 0
    for example in examples:
        if example.is_impossible:
            is_impossible_num += 1

    logger.info(f"Unsolvable example len: {is_impossible_num}")
