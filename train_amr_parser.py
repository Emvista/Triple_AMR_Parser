# built-in modules
import random
import os
import sys
from argparse import ArgumentParser
import warnings
from typing import List

# third-party modules
import transformers
from transformers import EarlyStoppingCallback
from transformers import MBartForConditionalGeneration
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from Seq2seqTokenizer import Seq2seqTokenizer
from load_data import load_seq2seq_data
from datasets import concatenate_datasets

# custom modules
from settings import PROJECT_DIR
AMR = os.path.join(PROJECT_DIR, "AMR")  # add AMR to the path
SMATCH = AMR + "/smatch"
sys.path.append(AMR)
from trainer.TripleSeq2seqTrainer import TripleSeq2seqTrainer
from settings import get_data_path, LANGUAGE_MAPPING
from utils import get_latest_checkpoint
import numpy as np
import torch
from datasets import interleave_datasets

def load_tokenizer(model_name:str, linearization_types:List=None, resume_from_checkpoint=None, output_dir=None):

    if args.resume_from_checkpoint:
        # if set to True, load the latest checkpoint
        # if set to a path, load the checkpoint from the path provided
        checkpoint_path = resume_from_checkpoint if isinstance(resume_from_checkpoint,str) else get_latest_checkpoint(output_dir)
        return AutoTokenizer.from_pretrained(checkpoint_path)

    # if not resuming from checkpoint, load the tokenizer from the model name and add special tokens
    else:
        special_tokens  = []
        if "penman" in linearization_types:
            special_tokens.append("penman")
        if "vannoord" in linearization_types:
            special_tokens.append("vnd")
        if "amr" in linearization_types:
            special_tokens.append("amr")

        try:
            print("loading fast tokenizer: ", model_name, file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        except: #error when loading jz trained models.. don't know why this happens but this fixes the pb for now
            print("loading tokenizer (non fast): ", model_name, file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        special_token_dict = {"additional_special_tokens" : special_tokens}
        tokenizer.add_special_tokens(special_token_dict, replace_additional_special_tokens=False)

        return tokenizer

# def update_lang_code_id(tokenizer, special_tokens):
#     # add special tokens for each language
#     for lang in special_tokens:
#         tokenizer.lang_code_to_id[lang] = tokenizer.convert_tokens_to_ids(lang)
#     return tokenizer


def filter_data(dataset, max_length:int, split:str):
    # filter data examples when it exceeds the max length
    filtered = dataset.filter(lambda x: len(x["labels"]) < max_length)
    print(f"{split} dataset: {len(dataset)-len(filtered)} examples filtered out of {len(dataset)}")
    return filtered


def get_train_file_suffix(without_variables, without_invrole):
    file_suffix = "tri"

    if without_variables and not without_invrole:
        file_suffix = "wo_var.tri"
    if without_invrole and not without_variables:
        file_suffix = "wo_invrole.tri"
    if without_variables and without_invrole:
        file_suffix = "wo_var.wo_invrole.tri"

    print("Training data suffix: ", file_suffix)
    return file_suffix

def main(args):

    # create output_dir if not exists
    output_dir = PROJECT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. load model and tokenizer
    # if resume_from_checkpoint is set, load the tokenizer from the checkpoint
    # for consistent tokenizing of evaluation dataset (bos token etc)
    # model is resumed from the checkpoint later in the code
    tokenizer = load_tokenizer(args.model_name, args.linearization_types, args.resume_from_checkpoint, output_dir)
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # 2. load dataset (load all tasks and then keep only the task that is used later)
    train_data_suffix = get_train_file_suffix(args.without_variables, args.without_invrole)
    task_file_suffix_dict = {"amr": train_data_suffix, "vannoord": "vnd", "penman": "penman"} # basic task for triples amr parsing
    
    train_data_path_dict = {}
    for lin_type in args.linearization_types:
        train_data_path_dict[lin_type] = get_data_path(args.task_name, split="train", suffix=task_file_suffix_dict[lin_type])
    dev_src, dev_tgt = get_data_path(args.task_name, split="eval", suffix="amr")
    test_src, test_tgt = get_data_path(args.task_name, split="test", suffix="amr")
    test_v3_src, test_v3_tgt = get_data_path(args.task_name, split="test_v3", suffix="amr")
    silver_test_src, silver_test_tgt = get_data_path(args.task_name, split="silvertest", suffix="amr")
    sub_train_src, sub_train_tgt = get_data_path(args.task_name, split="train_subset", suffix="amr")

    max_samples = 10 if args.debug_on else None

    train_datasets_dict = {}
    for task, (train_src, train_tgt) in train_data_path_dict.items():
        if task == args.generate_format:
            train_datasets_dict[task] = [load_seq2seq_data(train_src, train_tgt), "main"]  # main task
        else:
            train_datasets_dict[task] = [load_seq2seq_data(train_src, train_tgt), "auxiliary"]

    eval_dataset = load_seq2seq_data(dev_src, dev_tgt, max_samples)
    test_dataset = load_seq2seq_data(test_src, test_tgt, max_samples)
    test_v3_dataset = load_seq2seq_data(test_v3_src, test_v3_tgt, max_samples)
    silvertest_dataset = load_seq2seq_data(silver_test_src, silver_test_tgt, max_samples)
    train_subset = load_seq2seq_data(sub_train_src, sub_train_tgt, max_samples)

    # 2.1 preprocess the data
    src_lang, tgt_lang = args.task_name.split("-")
    seq2seq_tokenizer = Seq2seqTokenizer(tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, max_length=args.max_length)
    # tokenize the dataset

    for task, train_dataset_tuple in train_datasets_dict.items():
        # if "vnd" or "amr" is not in the vocab, it will be tagged as <unk>
        seq2seq_tokenizer.tokenizer.tgt_lang = LANGUAGE_MAPPING[task]
        train_datasets_dict[task][0] = train_dataset_tuple[0].map(seq2seq_tokenizer.tokenize, batched=True)

    # re-initialize the tokenizer tgt_lang to amr
    seq2seq_tokenizer.tokenizer.tgt_lang = tgt_lang
    eval_dataset = eval_dataset.map(seq2seq_tokenizer.tokenize, batched=True)
    test_dataset = test_dataset.map(seq2seq_tokenizer.tokenize, batched=True)
    test_v3_dataset = test_v3_dataset.map(seq2seq_tokenizer.tokenize, batched=True)
    silvertest_dataset = silvertest_dataset.map(seq2seq_tokenizer.tokenize, batched=True)
    train_subset = train_subset.map(seq2seq_tokenizer.tokenize, batched=True)

    # keep only the columns that are needed for training
    for task, train_dataset_tuple in train_datasets_dict.items():
        train_dataset_tuple[0].set_format(columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    test_v3_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    silvertest_dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    train_subset.set_format(columns=["input_ids", "attention_mask", "labels"])

    # concatenate all the datasets
    if len(args.linearization_types) == 1:
        train_dataset = list(train_datasets_dict.values())[0][0]
    elif len(args.linearization_types) == 2:
        probabilities = [args.main_task_probability if train_dataset_tuple[1] == "main" else round(1 - args.main_task_probability, 1)
                                     for task_name, train_dataset_tuple in train_datasets_dict.items()]
        train_dataset = interleave_datasets([train_dataset_tuple[0] for task_name, train_dataset_tuple in train_datasets_dict.items()],
                                            probabilities=probabilities,
                                            seed=args.random_seed,
                                            stopping_strategy="all_exhausted")

        print("Interleaved datasets with probabilities: ", probabilities)
        print("Train tasks: ", [task_name for task_name, train_dataset_tuple in train_datasets_dict.items()])

    # 2.2 create data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=seq2seq_tokenizer.tokenizer,
                                           model=model,
                                           padding='longest',
                                           max_length=args.max_length,
                                           return_tensors="pt")

    tgt_lang = LANGUAGE_MAPPING[args.generate_format]
    forced_bos_token_id = seq2seq_tokenizer.tokenizer.convert_tokens_to_ids(tgt_lang)

    training_args = Seq2SeqTrainingArguments(output_dir=output_dir,
                                             evaluation_strategy="steps",
                                             do_eval=args.do_eval,
                                             eval_steps=args.eval_steps,
                                             per_device_train_batch_size=args.per_device_train_batch_size,
                                             per_device_eval_batch_size=args.per_device_eval_batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             eval_accumulation_steps=args.eval_accumulation_steps,
                                             learning_rate=args.learning_rate,
                                             max_steps=args.max_steps,
                                             logging_strategy="steps",
                                             logging_steps=args.logging_steps,
                                             save_strategy="steps",
                                             save_steps=args.eval_steps,
                                             load_best_model_at_end=True, # if False=> last checkpoint, if True => best checkpoint
                                             lr_scheduler_type="linear",
                                             warmup_ratio=args.warmup_ratio,
                                             metric_for_best_model="smatch",
                                             greater_is_better=True,
                                             save_total_limit=args.save_total_limit,
                                             predict_with_generate=True,
                                             report_to=["wandb"],)

    # 4. train model
    trainer = TripleSeq2seqTrainer(model=model,
                                   tokenizer=seq2seq_tokenizer.tokenizer,
                                   data_collator=data_collator,
                                   args=training_args,
                                   train_dataset=train_dataset,
                                   eval_dataset=eval_dataset,
                                   callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.002)],
                                   train_subset=train_subset,
                                   without_variables=args.without_variables,
                                   no_wikify=args.no_wikify,
                                   forced_bos_token_id=forced_bos_token_id,
                                   generate_max_length=args.max_length,
                                   generate_num_beams=5,
                                   target_format=args.generate_format,
                                   eval_gold_file=dev_tgt.with_suffix(".pm"),
                                   test_gold_file=test_tgt.with_suffix(".pm"),
                                   test_v3_gold_file=test_v3_tgt.with_suffix(".pm"),
                                   silvertest_gold_file=silver_test_tgt.with_suffix(".pm"),
                                   sub_train_gold_file=sub_train_tgt.with_suffix(".pm")
                                   )

    if not args.no_training: # if training=True, either None or resume_from_checkpoint
        model.to(trainer.args.device)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    else:
        # only load the model from the checkpoint
        model = MBartForConditionalGeneration.from_pretrained(args.resume_from_checkpoint)
        model.to(trainer.args.device)
        trainer.model = model

    # Evaluate model on the test sets
    if args.test:
        trainer.predict(test_dataset, metric_key_prefix="test")
    if args.silvertest:
        trainer.predict(silvertest_dataset, metric_key_prefix="silvertest")


if __name__ == '__main__':

    import pprint

    args = ArgumentParser()
    args.add_argument("--model_name", type=str, default="facebook/mbart-large-50", help="model name")
    args.add_argument("--output_dir", type=str, default="amr_training", help="output directory")
    args.add_argument("--task_name", type=str, default="en-amr", help="task name, e.g. fr-amr for French AMR parsing")
    args.add_argument("--max_length", type=int, default=1024, help="maximum length of the input sequence")
    args.add_argument("--do_eval", type=bool, default=True, help="whether to evaluate the model during training or not")
    args.add_argument("--eval_steps", type=int, default=10, help="number of steps between evaluations")
    args.add_argument("--per_device_train_batch_size", type=int, default=2, help="batch size per device for training")
    args.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup ratio for the learning rate scheduler")
    args.add_argument("--per_device_eval_batch_size", type=int, default=2, help="batch size per device for evaluation")
    args.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of gradient accumulation steps")
    args.add_argument("--eval_accumulation_steps", type=int, default=1)
    args.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate for the model")
    args.add_argument("--max_steps", type=int, default=100, help="total number of training steps")
    args.add_argument("--logging_steps", type=int, default=None)
    args.add_argument("--save_total_limit", type=int, default=2)
    args.add_argument("--early_stopping_patience", type=int, default=15)
    args.add_argument("--resume_from_checkpoint", type=str, default=None, help="path to checkpoint e.g. training_outputs/amr_training/checkpoint-500")
    args.add_argument("--debug_on", action="store_true", help="use this flag to debug with shorter eval dataset")
    args.add_argument("--no_training", action="store_true", help="use this flag to only predict a trained model")
    args.add_argument("--without_variables", action="store_true", help="use this flag to train without variables (simplified graphs)")
    args.add_argument("--no_wikify", action="store_true", help="use this flag to skip wikification during training")
    args.add_argument("--without_invrole", action="store_true", help="use this flag to not use inverse roles in the training data (simplified graphs)")
    args.add_argument("--generate_format", type=str, choices=["amr", "vannoord", "penman"], required=True, help="generation format for evaluation (except for particular cases, it should match --linearization_types)")
    args.add_argument("--linearization_types", type=str, nargs='+', choices=["amr", "vannoord", "penman"], required=True, help="task types for the model, possible to add multiple types for multi-task training")
    args.add_argument("--test", action="store_true", help="use this flag to evaluate on the test set")
    args.add_argument("--silvertest", action="store_true", help="use this flag to evaluate on the silvertest set")
    args.add_argument("--random_seed", type=int, default=42)
    args.add_argument("--main_task_probability", type=float, default=0.5, help="probability of the main task in the training data")
    args = args.parse_args()

    # set seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    transformers.set_seed(args.random_seed)

    # set resume_from_checkpoint to False if not set
    if args.resume_from_checkpoint in [None, "False"] :
        print("resume_from_checkpoint is set to False")
        args.resume_from_checkpoint = False

    if args.resume_from_checkpoint== "True":
        print("resume_from_checkpoint is set to True")
        args.resume_from_checkpoint = True

    # set logging steps to eval steps if not set
    if args.logging_steps is None:
        args.logging_steps = args.eval_steps

    pprint.pp(args)

    # wandb logging setup
    if args.debug_on:
        os.environ["WANDB_MODE"] = "disabled" # to avoid logging to wandb for debugging
    else:
        os.environ["WANDB_PROJECT"] = "amr_triple_seq2seq"
        os.environ["WANDB__SERVICE_WAIT"] = "300"

    main(args)
