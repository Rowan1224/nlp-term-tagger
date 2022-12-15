
import pandas as pd
import os
import torch 
from torch import cuda
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import random
from sklearn.metrics import classification_report
from model import CRF
from dataloader import PreDataCollator

from args import create_arg_parser
from utils import  get_tag_mappings, compute_metrics_test, print_predictions, span_evaluation
os.environ["WANDB_DISABLED"] = "true"



### Seed all

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)





def main():


    args = create_arg_parser()
    device = 'cuda' if cuda.is_available() else 'cpu'
    # Load data as pandas dataframe
    test_df = pd.read_csv('./Dataset/test.csv')

    test_data = Dataset.from_pandas(test_df)




    tags_to_ids, ids_to_tags = get_tag_mappings()
    number_of_labels = len(tags_to_ids)



    MAX_LEN = args.seq_length
    MODEL_NAME = args.model_name
    CHECKPOINT = args.checkpoint
    IS_CRF = args.use_crf



    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


    collator = PreDataCollator(tokenizer=tokenizer, max_len=MAX_LEN, tags_to_ids = tags_to_ids)


    test_tokenized = test_data.map(collator, remove_columns=test_data.column_names, batch_size=4, num_proc=4, batched=True)





    if IS_CRF:

        saved_model_dir = f'./output/{MODEL_NAME}-CRF/checkpoint-{CHECKPOINT}/pytorch_model.bin'
        model = CRF(MODEL_NAME,ids_to_tags,number_of_labels,device=device)
        checkpoint = torch.load(saved_model_dir)
        model.load_state_dict(checkpoint)

    else:
        saved_model_dir = f'./output/{MODEL_NAME}-Base/checkpoint-{CHECKPOINT}'
        model = AutoModelForTokenClassification.from_pretrained(saved_model_dir, num_labels=number_of_labels)
        model = model.to(device)

    model = model.to(device)




    visualization = []
    acc = 0
    f1  = 0
    outputs = []

    all_true = []
    all_pred = []
    test_len = len(test_tokenized)

    _, ids_to_tags = get_tag_mappings()

    for i in tqdm(range(test_len)): 

        inp_ids = torch.as_tensor([test_tokenized[i]["input_ids"]]).to(device)
        label_ids = torch.as_tensor([test_tokenized[i]["labels"]]).to(device)
        
        mask = torch.as_tensor([test_tokenized[i]["attention_mask"]]).to(device)


        if IS_CRF:
            pred_ids = model(input_ids=inp_ids, attention_mask=mask, labels=label_ids)['logits']
        else:
            logits = model(input_ids=inp_ids, attention_mask=mask).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]
        
        result, predicts, tags = compute_metrics_test(pred_ids,label_ids,IS_CRF)
        
        all_true.extend(tags)
        all_pred.extend(predicts)

        
        pred_tags = [ids_to_tags[idx] for idx in predicts if idx!=-100]
        true_tags = [ids_to_tags[idx] for idx in tags if idx!=-100]
        
        vis, pred_tags, true_tags = print_predictions(test_data[i]['sent'],pred_tags,true_tags)
        
        outputs.append((test_data[i]['sent'], pred_tags, true_tags))
        
        acc += result['accuracy']
        f1 += result['f1']
        visualization.append(vis)
        
        
    print("#"*50)
    print('Token Level Evaluation')
    print("#"*50)
    print(classification_report(all_true,all_pred))
    print("#"*50)

    df = pd.DataFrame(outputs, columns=['sent','predictions','true'])
    true = df['true'].tolist()
    pred = df['predictions'].tolist()

    print("#"*50)
    print('Span Level Evaluation')
    print("#"*50)
    span_evaluation(true,pred)
    print("#"*50)


    
    model_type = 'CRF' if IS_CRF else 'Base'
    df.to_csv(f'./output/{MODEL_NAME}-{model_type}/outputs.csv',index=False)




if __name__ == '__main__':
    main()