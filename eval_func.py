import os
import timeit
import torch
import json
import logging
import numpy as np
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, SequentialSampler
from transformers.data.processors.squad import SquadResult

from tools.load_examples import load_and_cache_examples
from tools.eval_index import eval_f1_em, eval_auc, calc_roc_curve_index, calc_auc
from processors.postprocess import compute_predictions_logits
from tools.produce_ans import produce_answers, getQuesType
from tools.utils import to_list
from models.derived_models import BioModelClassify


logger = logging.getLogger("BIOBERT")

class SquadImpossible:

    def __init__(self, unique_id, impossible):
        self.impossible = impossible
        self.unique_id = unique_id


def eval_one_batch(model, batch, features): # 确保model.eval()也被执行
    batch_result = []
    batch_pred_unsolvable = []

    with torch.no_grad():
        feature_indices = batch[3]
        outputs = model(batch, is_training=False) # 如果出现missing position 'batch', 那就是忘了设置single_gpu

    for i, feature_index in enumerate(feature_indices):
        eval_feature = features[feature_index.item()]
        unique_id = int(eval_feature.unique_id)

        if isinstance(model, BioModelClassify): # USE_CLASSIFY?
            batch_pred_unsolvable.append(SquadImpossible(unique_id, outputs['pred_impossible'][i]))

        output_list = [to_list(outputs['start_logits'][i]), to_list(outputs['end_logits'][i])]

        # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
        # models only use two.
        if len(output_list) >= 5:
            start_logits = output_list[0]
            start_top_index = output_list[1]
            end_logits = output_list[2]
            end_top_index = output_list[3]
            cls_logits = output_list[4]

            result = SquadResult(
                unique_id,
                start_logits,
                end_logits,
                start_top_index=start_top_index,
                end_top_index=end_top_index,
                cls_logits=cls_logits,
            )

        else:
            start_logits, end_logits = output_list
            result = SquadResult(unique_id, start_logits, end_logits)

        batch_result.append(result)

    return batch_result, batch_pred_unsolvable


def eval_by_model(args, model, eval_dataloader, features, tqdm_enabled=True):
    # Eval!
    all_results = []
    all_pred_unsolvable = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not tqdm_enabled):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        batch_result, batch_pred_unsolvable = eval_one_batch(model, batch, features)
        all_results += batch_result
        all_pred_unsolvable += batch_pred_unsolvable

    return all_results, all_pred_unsolvable


def evaluate(args, model, tokenizer, prefix=""):
    USE_CLASSIFY = isinstance(model, BioModelClassify)
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num features = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    start_time = timeit.default_timer()

    all_results, all_pred_unsolvable = eval_by_model(args, model, eval_dataloader, features)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per feature)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None
    
    ques_type = {example.qas_id: getQuesType(example.question_text)  for example in examples}
    answers = {example.answers['id']: example.answers['answers'] for example in examples}

    if args.calc_auc:
        offset = args.threshold_step
        threshold_range = np.arange(args.threshold_start, args.threshold_end + offset, offset)
    else:
        threshold_range = [args.null_score_diff_threshold]
    
    tpr = []
    fpr = []

    # for threshold in range(0, offset, 1 + offset):
    # for threshold in np.arange(40, -40 + offset, offset): # 这个计算出来的AUC是负的
    for threshold in threshold_range:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            threshold,
            tokenizer,
            all_pred_unsolvable if USE_CLASSIFY else None,
        )

        # if args.calc_auc:
        eval_rst = eval_auc(answers, predictions, ques_type, output_dir=args.output_dir, prefix=prefix, threshold=str(threshold))
        # else:
            # eval_rst = eval_f1_em(answers, predictions)

        logger.info("** Evaluation Info ************************************")
        logger.info(f"   Test samples num: {len(examples)}")

        logger.info("** Evaluation Results ************************************")
        logger.info(f"  TOTAL EM = {eval_rst['exact_match']:.2f}")
        logger.info(f"  TOTAL F1 score = {eval_rst['f1']:.2f}")
        logger.info(f"  SOLVABLE EM = {eval_rst['solvable_em']:.2f}")
        logger.info(f"  SOLVABLE F1 score = {eval_rst['solvable_f1']:.2f}")
        logger.info(f"** THRESHOLD = {threshold} ************************************")
        logger.info(f"  TP = {eval_rst['tp']}")
        logger.info(f"  FP = {eval_rst['fp']}")
        logger.info(f"  TN = {eval_rst['tn']}")
        logger.info(f"  FN = {eval_rst['fn']}")
        cur_tpr, cur_fpr = calc_roc_curve_index(eval_rst['tp'], eval_rst['fp'], eval_rst['tn'], eval_rst['fn'])
        tpr.append(cur_tpr)
        fpr.append(cur_fpr)
        logger.info(f"  CURRENT TPR = {cur_tpr}")
        logger.info(f"  CURRENT FPR = {cur_fpr}")
        
    if args.calc_auc:
        with open(os.path.join(args.output_dir, 'auc_index_{}.json'.format(prefix)), 'w') as fh:
            tmp = {"TPR": tpr, "FPR": fpr}
            json.dump(tmp, fh)
        auc = calc_auc(np.array(tpr), np.array(fpr))
        logger.info(f"** AUC ************************************")
        logger.info(f"  AUC = {auc}")

    with open(os.path.join(args.output_dir, "predictions_{}.html".format(prefix)), 'w', encoding='utf-8') as fh:
        with open(output_prediction_file, 'r') as reader:
            pre_json = json.load(reader)
            f1_scores = eval_rst['all_f1']
            for idx, example in enumerate(examples):
                produce_answers(pre_json[example.qas_id], example, f1_scores[idx], fh)    

    return {
        "EM": eval_rst["exact_match"], 
        "F1_score": eval_rst["f1"], 
        "Solvable_EM": eval_rst["solvable_em"], 
        "Solvable_F1": eval_rst["solvable_f1"]
    }