
import evaluate
import torch
from colorama import Fore, Style, Back
import re

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")


def compute_metrics_test(preds,labels, is_crf=False):
    

    tr_active_acc = labels != -100
    
    if is_crf:
        pr_active_acc = preds != -100
        predicts = torch.masked_select(preds, pr_active_acc)
    else:
        predicts = torch.masked_select(preds, tr_active_acc)

    tags = torch.masked_select(labels, tr_active_acc)
    
    acc = metric_acc.compute(predictions=predicts, references=tags)
    f1 = metric_f1.compute(predictions=predicts, references=tags, average='macro')
    
    return {'accuracy': acc['accuracy'], 'f1':f1['f1']}, tags.tolist(), predicts.tolist()

def get_tag_mappings():
    
    unique_tags = ['O','I','B']
    tags_to_ids = {k: v for v, k in enumerate(unique_tags)}
    ids_to_tags = {v: k for v, k in enumerate(unique_tags)}

    return tags_to_ids, ids_to_tags



    
def match(t,p):
    
    trueSpans = set(get_span(t))
    predSpans = set(get_span(p))
    
    intersection = trueSpans.intersection(predSpans)

    return len(trueSpans), len(predSpans), len(intersection)


def span_evaluation(true, pred):

    totalGold = 0
    totalPred = 0
    correct = 0
    for t,p in zip(true,pred):
        
        trueSpanCount, predSpanCount, matchCount = match(t,p)
        totalGold += trueSpanCount
        totalPred += predSpanCount
        correct+= matchCount

    recall = correct/totalGold
    precision = correct/totalPred
    f1 = 2/((1/recall)+(1/precision))

    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1: {f1:.2f}")



def print_predictions(tokens, pred_tags, true_tags):
    
    tokens = tokens.split()

    if len(tokens) != len(pred_tags):
        print(tokens)
        return " "
    
    output = []
    
    
    for t,tl,pl in zip(tokens,true_tags,pred_tags):

        if tl == pl:
            o = f"{t} {Back.GREEN}[{tl}][{pl}]{Style.RESET_ALL}"

        else:
            o = f"{t} {Back.GREEN}[{tl}]{Style.RESET_ALL}{Back.RED}[{pl}]{Style.RESET_ALL}"

        output.append(o)
        
    return " ".join(output)," ".join(pred_tags), " ".join(true_tags)


def get_span(label):

    pattern = "B( I)+|B|B O( I)"
    matches = re.finditer(pattern, label)
    spansPositions = [match.span() for match in matches]


    return spansPositions


def compute_metrics_crf(pred):
    

    pred_logits = pred.predictions
    pred_ids = torch.tensor(pred_logits)

    tr_active_acc = torch.from_numpy(pred.label_ids != -100)
    pr_active_acc = torch.from_numpy(pred_logits != -100)

    train_tags = torch.masked_select(torch.from_numpy(pred.label_ids), tr_active_acc)
    train_predicts = torch.masked_select(pred_ids, pr_active_acc)

    acc = metric_acc.compute(predictions=train_predicts, references=train_tags)
    f1 = metric_f1.compute(predictions=train_predicts, references=train_tags, average='macro')
    
    return {'accuracy': acc['accuracy'], 'f1':f1['f1']}


def compute_metrics(pred):
    
    
    pred_logits = pred.predictions
    pred_ids = torch.from_numpy(np.argmax(pred_logits, axis=-1))

    tr_active_acc = torch.from_numpy(pred.label_ids != -100)

    train_tags = torch.masked_select(torch.from_numpy(pred.label_ids), tr_active_acc)
    train_predicts = torch.masked_select(pred_ids, tr_active_acc)

    acc = metric_acc.compute(predictions=train_predicts, references=train_tags)
    f1 = metric_f1.compute(predictions=train_predicts, references=train_tags, average='macro')
    
    return {'accuracy': acc['accuracy'], 'f1':f1['f1']}
