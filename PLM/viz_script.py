import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Style, Back
from utils import get_span
import numpy as np
from sklearn import metrics
from collections import Counter

#TODO: MULTICLASS MATRIX https://aihints.com/seaborn-confusion-matrix-plot-confusion-matrix-in-python/

def bar_plot(dev, test, train):
    datasets = {"Train": train, "Dev": dev, "Test": test}
    viz_labels_df = pd.DataFrame.from_dict(datasets, orient="index")
    viz_labels_df["Dataset"] = viz_labels_df.index
    print(viz_labels_df)
    df = pd.melt(viz_labels_df, id_vars="Dataset", var_name="Tags", value_name="Count")
    print(df)
    fig = sns.catplot(x='Dataset', y='Count', hue='Tags', data=df, kind='bar', palette="pastel")
    fig.set(title='Tag Distribution')
    fig.set_style("whitegrid")
    fig.savefig("Distribution of Tags.png", bbox_inches="tight")

def dataset_explorer(path):

    with open(path, 'r') as f:
        sent_labels = f.readlines()
    
    sent_labels = [lab.split() for lab in sent_labels]
    sent_labels = [label.upper() for sent in sent_labels for label in sent]  # test and train had lower cases
    return Counter(sent_labels)

def print_predictions(tokens, pred_tags, true_tags):
    
    tokens = tokens.split()
    pred_tags = pred_tags.split()
    true_tags = true_tags.split()
    
    if len(tokens) != len(pred_tags):
        return " "
    
    output = []
    for t,tl,pl in zip(tokens,true_tags,pred_tags):

        if tl == pl:
            o = f"{t} {Back.GREEN}[{tl}][{pl}]{Style.RESET_ALL}"

        else:
            o = f"{t} {Back.GREEN}[{tl}]{Style.RESET_ALL}{Back.RED}[{pl}]{Style.RESET_ALL}"

        output.append(o)
        
    return " ".join(output)

def span_counter(true, mod_1_pred, mod_2_pred):

    both_incorrect = 0
    m1_only = 0
    m2_only = 0
    both_correct = 0
    true_span_idx = get_span(true)
    mod_1_span_idx = get_span(mod_1_pred)
    mod_2_span_idx = get_span(mod_2_pred)
    mod_1_res = []
    mod_2_res = []
    for span in true_span_idx:
        if (span in mod_1_span_idx) and (span in mod_2_span_idx):
            both_correct += 1
            mod_1_res.append(1)
            mod_2_res.append(1)
        elif (span in mod_1_span_idx) and (span not in mod_2_span_idx):
            m1_only += 1
            mod_1_res.append(1)
            mod_2_res.append(0)
        elif (span not in mod_1_span_idx) and (span in mod_2_span_idx):
            m2_only += 1
            mod_1_res.append(0)
            mod_2_res.append(1)
        #elif (span not in mod_1_span_idx) and (span not in mod_2_span_idx):
        else:
            both_incorrect += 1
            mod_1_res.append(0)
            mod_2_res.append(0)
            #raise Exception("weird comparison")
    qual_results = [both_correct, m1_only, m2_only, both_incorrect]
    return qual_results, mod_1_res, mod_2_res
    #return both_correct, m1_only, m2_only, both_incorrect, len(true_span_idx)

def output_reader(path):
    df = pd.read_csv(path)
    df['visualizations'] = df.apply(lambda x: print_predictions(x['sent'], x['predictions'], x["true"]), axis=1)
    return df

def main():
    
    dev_path = "../Dataset/dev/labels.txt"
    train_path = "../Dataset/train/labels.txt"
    test_path = "../Dataset/test/labels.txt"
    dev = dataset_explorer(dev_path)
    train = dataset_explorer(train_path)
    test = dataset_explorer(test_path)
    bar_plot(dev, test, train)

    bert_base = output_reader("../outputs_for_viz/bert_outputs.csv")
    crf_bert = output_reader("../outputs_for_viz/outputs-bert-crf.csv")
    distilbert_base = output_reader("../outputs_for_viz/distil_outputs.csv")
    crf_distilbert = output_reader("../outputs_for_viz/outputs-distill-crf.csv")
    
    lstm_base = output_reader("../outputs_for_viz/outputs-Base-LSTM.csv")
    lstm_crf = output_reader("../outputs_for_viz/outputs-CRF-LSTM.csv")

    #print(bert_base_viz.iloc[0])
    #print()
    #print(crf_bert_viz.iloc[0])

    test_1 = bert_base["visualizations"].iloc[0]
    test_2 = distilbert_base["visualizations"].iloc[0]

    span_sents = dict()
    poor_annotations = []
    overall_mod_1 = []
    overall_mod_2 = []
    for i in range(bert_base.shape[0]):
        #pred_1 = bert_base["predictions"].iloc[i]
        #pred_1 = distilbert_base["predictions"].iloc[i]
        pred_1 = lstm_base["predictions"].iloc[i]
        #pred_2 = crf_distilbert["predictions"].iloc[i]
        #pred_2 = crf_bert["predictions"].iloc[i]
        pred_2 = lstm_crf["predictions"].iloc[i]
        #true = distilbert_base["true"].iloc[i]
        #true = bert_base["true"].iloc[i]
        true = lstm_base["true"].iloc[i]
        #sent = bert_base["sent"].iloc[i]
        qual_results, mod_1_res, mod_2_res = span_counter(true, pred_1, pred_2)
        overall_mod_1.extend(mod_1_res)
        overall_mod_2.extend(mod_2_res)
        both_corr, m1_only, m2_only, both_incorr = qual_results
        if (both_corr + m1_only + m2_only + both_incorr) == 0:
            poor_annotations.append(i)
        span_sents.setdefault(f"sent_{i}", {"both_corr": both_corr, "m1_only": m1_only,
                                            "m2_only": m2_only, "both_incorr": both_incorr})
    
    
    spans = pd.DataFrame.from_dict(span_sents, orient="index")
    #spans.to_csv("BERT_and_CRF.csv")
    #spans.to_csv("DistilBERT_and_CRF.csv")
    #spans.to_csv("LSTM_and_CRF.csv")
    print(spans.head())
    # Select Confusion Matrix Size
    results = [[spans["both_corr"].sum(), spans["m1_only"].sum()],
               [spans["m2_only"].sum(), spans["both_incorr"].sum()]]
    results = np.array(results)
    plt.figure(figsize=(10, 8))
    
    # Create Confusion Matrix
    x_axis_labels = ["Model 2 Correct", "Model 2 Incorrect"]
    y_axis_labels = ["Model 1 Correct", "Model 2 Incorrect"]
    
    # Set names to show in boxes
    classes = ["True Positive","False Negative","False Positive","True Negative"]
    
    # Set values format
    values = ["{0:0.0f}".format(x) for x in results.flatten()]
    
    # Find percentages and set format
    percentages = ["{0:.1%}".format(x) for x in results.flatten()/np.sum(results)]
    
    # Combine classes, values and percentages to show 
    combined = [f"{i}\n{j}\n{k}" for i, j, k in zip(classes, values, percentages)]
    combined = np.asarray(combined).reshape(2,2)


    #sns_confusion = sns.heatmap(results, annot=True, fmt='d', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    sns_confusion = sns.heatmap(results, annot=combined, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="YlGnBu", fmt="")
    # Set the Title
    sns_confusion.set(title='LSTM Confusion Matrix')
    #sns_confusion.set(title='BERT Confusion Matrix')
    #sns_confusion.set(title='DistilBERT Confusion Matrix')
    
    # Set the Labels
    sns_confusion.set(xlabel='LSTM-CRF', ylabel='LSTM Base')
    #sns_confusion.set(xlabel='BERT-CRF', ylabel='BERT Base')
    #sns_confusion.set(xlabel='DistilBERT-CRF', ylabel='DistilBERT Base')
    
    # Display the Confusion Matrix
    #plt.show()
    plt.savefig("LSTM Confusion Matrix")
    #plt.savefig("BERT Confusion Matrix")
    #plt.savefig("DistilBERT Confusion Matrix")
    

    #confusion_matrix = metrics.confusion_matrix(overall_mod_1, overall_mod_2)
    #print(confusion_matrix)
    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    #cm_display.plot()
    #plt.savefig("mygraph.png")
    #print(poor_annotations)
    #for i in poor_annotations:
    #    #print(bert_base["true"].iloc[i])
    #    continue
    print(bert_base["sent"].iloc[28])
    print(bert_base["true"].iloc[28])
    print(bert_base["sent"].iloc[98])
    print(bert_base["true"].iloc[98])
    #plt.show() 
    #for t_1, t_2 in zip(test_1, test_2):
    #for t_1, t_2 in zip(test_1, test_2):
    #    if t_1 != t_2:
    #        print(t_1)


if __name__ == '__main__':
    main()