
import pandas as pd
import os
import torch 
from torch import cuda
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
import random
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import evaluate

from model import CRF
from dataloader import PreDataCollator
from args import create_arg_parser
from utils import get_tag_mappings, compute_metrics, compute_metrics_crf

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"



metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

### Seed all

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



def main():

    args = create_arg_parser()
    device = 'cuda' if cuda.is_available() else 'cpu'

    train_df = pd.read_csv('./Dataset/train.csv')
    dev_df = pd.read_csv('./Dataset/dev.csv')


    train_data = Dataset.from_pandas(train_df)
    dev_data = Dataset.from_pandas(dev_df)


    tags_to_ids, ids_to_tags = get_tag_mappings()
    number_of_labels = len(tags_to_ids)



    MAX_LEN = args.seq_length
    MODEL_NAME = args.model_name
    IS_CRF = args.use_crf

    EPOCHS = args.epoch
    LEARNING_RATE = args.learning_rate
    TRAIN_BATCH_SIZE = args.batch
    SAVE_STEPS = 50
    EVAL_STEPS = 50
    SAVE_LIMIT = None
    WARMUP_STEPS = 2


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


    collator = PreDataCollator(tokenizer=tokenizer, max_len=MAX_LEN, tags_to_ids = tags_to_ids)


    train_tokenized = train_data.map(collator, remove_columns=train_data.column_names, batch_size=4, num_proc=4, batched=True)
    dev_tokenized = dev_data.map(collator, remove_columns=dev_data.column_names, batch_size=4, num_proc=4, batched=True)


    if IS_CRF:
        model = CRF(MODEL_NAME,ids_to_tags,number_of_labels,device=device)
        
    else:
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME,num_labels=number_of_labels)
    
    model = model.to(device)
    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')

    

    training_args = TrainingArguments(
    output_dir=f"./output/{MODEL_NAME}-CRF" if IS_CRF else f"./output/{MODEL_NAME}-Base",
    group_by_length=True,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=EPOCHS,
    fp16=False,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=EVAL_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    save_total_limit=SAVE_LIMIT,
    )




    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics_crf if IS_CRF else compute_metrics,
        train_dataset=train_tokenized,
        eval_dataset=dev_tokenized,
        tokenizer=tokenizer
    )


    trainer.train()

if __name__ == '__main__':
    main()
