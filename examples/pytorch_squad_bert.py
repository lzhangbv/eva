# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import os
import random
import math
import timeit
import time
import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')

strhdlr = logging.StreamHandler()
strhdlr.setFormatter(formatter)
logger.addHandler(strhdlr) 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SPEED = False

# for bert
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import kfac
import kfac.backend as backend
import horovod.torch as hvd
from utils import *

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def initialize():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='bert-base-uncased',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--log_dir",
        default='./logs',
        type=str
    )

    # K-FAC parameters
    parser.add_argument('--kfac-name', type=str, default='inverse', 
                        help='choises: %s' % kfac.kfac_mappers.keys() + ', default: '+'inverse')
    parser.add_argument('--exclude-parts', type=str, default='',
                        help='choises: CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor')
    parser.add_argument('--kfac-update-freq', type=int, default=0,
                        help='iters between kfac inv ops (0 = no kfac) (default: 0)')
    parser.add_argument('--kfac-cov-update-freq', type=int, default=1,
                        help='iters between kfac cov ops (default: 1)')
    parser.add_argument('--stat-decay', type=float, default=0.95,
                        help='Alpha value for covariance accumulation (default: 0.95)')
    parser.add_argument('--damping', type=float, default=0.002,
                        help='KFAC damping factor (default 0.003)')
    parser.add_argument('--kl-clip', type=float, default=0,
                        help='KL clip (default: 0)')
    parser.add_argument('--use-adamw', type=int, default=1, 
                        help='use AdamW and its corresponding learnign rate schedule (default: 1)')

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int, 
            help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=0, type=int, 
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

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

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )


    # Setup CUDA, GPU & distributed training
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    hvd.init()
    backend.init("Horovod")

    args.local_rank = hvd.local_rank()

    # Set seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True    

    # Logging Settings
    os.makedirs(args.log_dir, exist_ok=True)
    if args.kfac_update_freq > 0:
        alg = args.kfac_name
    else:
        alg = 'adamw' if args.use_adamw else 'sgd'
    
    logfile = os.path.join(args.log_dir,
        'squad_{}_ep{}_bs{}_gpu{}_kfac{}_{}.log'.format(args.model_type, 
        args.num_train_epochs, args.per_gpu_train_batch_size, hvd.size(), 
        args.kfac_update_freq, alg))

    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 

    args.verbose = True if hvd.rank() == 0 else False
    
    if args.verbose:
        logger.info(args)
    
    return args

def preprocess_and_cache(args, tokenizer, evaluate, cached_features_file):
    if hvd.size() > 1 and hvd.rank() > 0:
        # Make sure only the first process in distributed training process the dataset.
        hvd.barrier()

    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
    # Make sure the SQuAD dataset has been downloaded and saved in data_dir/xxx_file.
    if evaluate:
        examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
    else:
        examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

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
    
    logger.info("Saving features into cached file %s", cached_features_file)
    torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if hvd.size() > 1 and hvd.rank() == 0:
        hvd.barrier()

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    """Load data features from cache or dataset file"""

    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Preprocess dataset if the cache does not exist
    if not os.path.exists(cached_features_file):
        preprocess_and_cache(args, tokenizer, evaluate, cached_features_file)

    # Load features and dataset from cache
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
    )

    if output_examples:
        return dataset, examples, features
    return dataset

def build_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=None,
        use_fast=False
    )
    return tokenizer

def get_dataset(args):
    # load pretrained tokenizer
    tokenizer = build_tokenizer(args)
    
    # load train datasets
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
    eval_dataset, eval_examples, eval_features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    test_inputs = (eval_examples, eval_features, tokenizer) # used to calculate F1 score

    # todo: train and valid split, used to calculate validation loss
    valid_size = int(len(train_dataset) * 0.005)  #0.005 takes ~5s
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    
    # train dataloader
    args.batch_size = args.per_gpu_train_batch_size
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    # valid dataloader
    args.eval_batch_size = args.per_gpu_eval_batch_size
    valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.eval_batch_size, sampler=valid_sampler, **kwargs)
    
    # eval dataloader
    args.eval_batch_size = args.per_gpu_eval_batch_size
    test_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    test_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=args.eval_batch_size, sampler=test_sampler, **kwargs)

    return train_sampler, train_loader, valid_loader, test_loader, test_inputs


def get_model(args):
    # Load pretrained model
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=None,
    )
    
    vocab_size = config.vocab_size # vocab_size=30522
    #vocab_size = 2 # for qa output layer

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=None,
    )
    
    #if args.verbose:
    #    logger.info("Model:%s" % model)

    if args.cuda:
        model.cuda()

    # optimizer
    criterion = nn.CrossEntropyLoss()
    if args.kfac_update_freq == 0:
        if args.use_adamw:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            # use Adam
            #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09)
        else:
            args.learning_rate = args.learning_rate * hvd.size()
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                    momentum=0.9, weight_decay=args.weight_decay)
        preconditioner = None
    else:
        args.learning_rate = args.learning_rate * hvd.size()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                momentum=0.9, weight_decay=args.weight_decay)
        KFAC = kfac.get_kfac_module(args.kfac_name)
        
        preconditioner = KFAC(
                model, lr=args.learning_rate, 
                factor_decay=args.stat_decay,
                damping=args.damping, kl_clip=args.kl_clip,
                fac_update_freq=args.kfac_cov_update_freq,
                kfac_update_freq=args.kfac_update_freq,
                exclude_parts=args.exclude_parts,
                exclude_vocabulary_size=vocab_size)

    optimizer = hvd.DistributedOptimizer(optimizer, 
                                        named_parameters=model.named_parameters(),
                                        op=hvd.Average,
                                        backward_passes_per_step=1)
    if hvd.size() > 1:
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    return model, optimizer, preconditioner

def train(args, epoch, model, optimizer, preconditioner, train_sampler, train_loader, validate, valid_loader):
    model.train()
    train_sampler.set_epoch(epoch)

    train_loss = Metric('train_loss')
    display = 10
    avg_time = 0.0
    tot_time = 0.0
    ittimes = []
    for batch_idx, batch in enumerate(train_loader):
        if args.cuda:
            batch = tuple(t.cuda() for t in batch)
        optimizer.zero_grad()
        
        stime = time.time()
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        outputs = model(**inputs)
        loss = outputs[0]

        with torch.no_grad():
            train_loss.update(loss)
        
        loss.backward()
        optimizer.synchronize()

        if preconditioner is not None:
            preconditioner.step(epoch=epoch)
        
        with optimizer.skip_synchronize():
            optimizer.step()
        
        args.global_step += 1
        avg_time += (time.time()-stime)

        if (batch_idx + 1) % display == 0:
            if args.verbose:
                if SPEED:
                    logger.info("[%d][%d] time: %.3f, speed: %.3f examples/s" % (epoch, batch_idx, avg_time/display, args.batch_size/(avg_time/display)))
                logger.info("[%d] epoch [%d] iteration train loss: %.4f" % (epoch, batch_idx, train_loss.avg.item()))
                train_loss = Metric('train_loss')
            
            if validate and hvd.rank() == 0:
                valid_loss = cal_validate_loss(model, valid_loader, args)
                logger.info("[%d] epoch [%d] iteration valid loss: %.4f" % (epoch, batch_idx, valid_loss))
                tot_time += avg_time
                #logger.info("Total time: %.2f, validation loss: %.4f" % (tot_time, valid_loss))
            
            ittimes.append(avg_time/display)
            avg_time = 0.0

        if batch_idx >= 60 and SPEED: # for average iteration time
            if args.verbose:
                logger.info("Iteration time: mean %.3f, std: %.3f" % (np.mean(ittimes[1:]),np.std(ittimes[1:])))
            break
        
        #if tot_time > 200: # for early optimization comparison
        #    break

        if args.max_steps > 0 and args.global_step > args.max_steps:
            break
        

def cal_validate_loss(model, valid_loader, args):
    loss_sum = 0.0
    loss_cnt = 0
    eval_batch_num = -1

    for batch_idx, batch in enumerate(valid_loader):
        if args.cuda:
            batch = tuple(t.cuda() for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            # get valid loss during training process
            loss = outputs[0]

            loss_sum += loss.item()
            loss_cnt += batch[0].shape[0]

        if eval_batch_num > 0 and batch_idx >= eval_batch_num:
            break

    return loss_sum / loss_cnt


def cal_evaluate_F1_score(model, test_loader, test_inputs, args):
    model.eval()
    examples, features, tokenizer = test_inputs[0], test_inputs[1], test_inputs[2]

    all_results = []
    for batch in test_loader:
        if args.cuda:
            batch = tuple(t.cuda() for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            feature_indices = batch[3]

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            #output = [to_list(output[i]) for output in outputs]
            output = [to_list(output[i]) for output in outputs.to_tuple()]

            assert len(output) == 2 # check whether two arguments used for predictions
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    # compute predictions
    output_prediction_file = os.path.join(args.log_dir, "squad_bert_predictions.json")
    output_nbest_file = os.path.join(args.log_dir, "squad_bert_nbest_predictions.json")

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        None, #output_prediction_file,
        None, #output_nbest_file,
        None, #output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    args = initialize()
    args.global_step = 0

    train_sampler, train_loader, valid_loader, test_loader, test_inputs = get_dataset(args)

    model, optimizer, preconditioner = get_model(args)
    valid_in_progress = True
    eval_in_progress = True

    start = time.time()
    for epoch in range(args.num_train_epochs):
        train(args, epoch, model, optimizer, preconditioner, 
                train_sampler, train_loader, valid_in_progress, valid_loader)
        
        if eval_in_progress and hvd.rank() == 0:
            results = cal_evaluate_F1_score(model, test_loader, test_inputs, args)
            logger.info("Epoch [%d] F1 score: %.4f" % (epoch, results['f1']))
        
        if args.max_steps > 0 and args.global_step > args.max_steps:
            break

    if args.verbose:
        logger.info("Total Training Time: %s", str(datetime.timedelta(seconds=time.time() - start)))
    
    if not eval_in_progress and hvd.rank() == 0:
        results = cal_evaluate_F1_score(model, test_loader, test_inputs, args)
        logger.info("Evaluation exact match: %.4f, F1 score: %.4f" % (results['exact'], results['f1']))
