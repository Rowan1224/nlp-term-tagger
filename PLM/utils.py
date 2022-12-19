
import evaluate
import torch
from colorama import Fore, Style, Back
import re
import numpy as np

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")


def compute_metrics_test(preds,labels, is_crf=False):
    """
    It takes in the predictions and labels, and returns the accuracy and f1 score
    
    :param preds: the predictions from the model
    :param labels: the actual labels
    :param is_crf: whether the model is a CRF or not. If it is, then the predictions are the tag
    indices, and we need to convert them to the actual tags, defaults to False (optional)
    """
    

    tr_active_acc = labels != -100
    
    if is_crf:
        pr_active_acc = preds != -100
        predicts = torch.masked_select(preds, pr_active_acc)
    else:
        predicts = torch.masked_select(preds, tr_active_acc)

    tags = torch.masked_select(labels, tr_active_acc)
    
    acc = metric_acc.compute(predictions=predicts, references=tags)
    f1 = metric_f1.compute(predictions=predicts, references=tags, average='macro')
    
    return {'accuracy': acc['accuracy'], 'f1':f1['f1']}, predicts.tolist(), tags.tolist()

def get_tag_mappings():
    """
    It creates a dictionary that maps each tag to a unique integer, and another dictionary that maps
    each unique integer to a tag
    :return: A dictionary of tags to ids and a dictionary of ids to tags.
    """
    
    unique_tags = ['O','I','B']
    tags_to_ids = {k: v for v, k in enumerate(unique_tags)}
    ids_to_tags = {v: k for v, k in enumerate(unique_tags)}

    return tags_to_ids, ids_to_tags



    
def match(t,p):
    """
    It takes two strings, `t` and `p`, and returns the number of spans in `t`, the number of spans in
    `p`, and the number of spans that are in both `t` and `p`
    
    :param t: the true spans
    :param p: predicted spans
    :return: The number of true spans, the number of predicted spans, and the number of spans that are
    in both the true and predicted sets.
    """
    
    trueSpans = set(get_span(t))
    predSpans = set(get_span(p))
    
    intersection = trueSpans.intersection(predSpans)

    return len(trueSpans), len(predSpans), len(intersection)


def span_evaluation(true, pred):
    """
    For each true and predicted span, count the number of true spans, predicted spans, and matches.
    Then, calculate recall, precision, and F1.
    
    :param true: a list of lists of tuples, where each tuple is a span of text
    :param pred: the predicted spans
    """

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
    """
    It takes in a sentence, the predicted tags and the true tags and prints out the sentence with the
    tags in color
    
    :param tokens: the list of tokens
    :param pred_tags: the predicted tags
    :param true_tags: the true tags for the sentence
    """
    
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
    """
    It takes a string of labels and returns a list of tuples, where each tuple is a span of the labels
    
    :param label: the label of the sentence
    :return: A list of tuples, each tuple is a span of a match.
    """

    pattern = "B( I)+|B|B O( I)"
    matches = re.finditer(pattern, label)
    spansPositions = [match.span() for match in matches]


    return spansPositions


def compute_metrics_crf(pred):
    """
    > We're going to compute the accuracy and f1 score of the predictions made by the model
    
    :param pred: the output of the model
    """
    

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
    """
    It takes the predictions from the model and computes the accuracy and f1 score
    
    :param pred: the prediction object returned by the model
    """
    
    
    pred_logits = pred.predictions
    pred_ids = torch.from_numpy(np.argmax(pred_logits, axis=-1))

    tr_active_acc = torch.from_numpy(pred.label_ids != -100)

    train_tags = torch.masked_select(torch.from_numpy(pred.label_ids), tr_active_acc)
    train_predicts = torch.masked_select(pred_ids, tr_active_acc)

    acc = metric_acc.compute(predictions=train_predicts, references=train_tags)
    f1 = metric_f1.compute(predictions=train_predicts, references=train_tags, average='macro')
    
    return {'accuracy': acc['accuracy'], 'f1':f1['f1']}
