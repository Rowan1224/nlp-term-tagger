from alignment import AnnotatedDataset
from models import DistilbertNER, CRFDistilBERT
from train_eval import NERTrainer, NEREvaluation, create_raw_data
from datasets import Dataset
from torch.utils.data import DataLoader
#from deeper_ner_eval import compute_metrics
import sys
import torch  # using torch 1.12.1
import pandas as pd
from crf_train_eval import CustomTrainer
from sklearn.metrics import classification_report
#import tqdm


def init_crfbert(ids_to_labels, unique_labels):
    crf_distilbert = CRFDistilBERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"  # allennlp and torch incompatible with my GPU
    model = crf_distilbert(ids_to_labels, len(unique_labels), device=device)
    return model.to(device)


def main(annotation_files, type_model):
    """
    opens file, tokenizes and finds unique labels before feeding data to main class
    """


    unique_labels, train_sents, train_labels = create_raw_data(annotation_files[0])  # algined_labels_760
    _, valid_sents, valid_labels = create_raw_data(annotation_files[1])  # ignore unique labels from val, they are both the same
    _, test_sents, test_labels = create_raw_data(annotation_files[2])  # ignore unique labels from val, they are both the same
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}                
    
    if type_model == "crf_distilbert":
        model = init_crfbert(ids_to_labels, unique_labels)

        output_path = "./output_model"
        crf_train_sents = [' '.join(sent) for sent in train_sents]
        crf_valid_sents = [' '.join(sent) for sent in valid_sents]
        crf_train_labels = [' '.join(labels) for labels in train_labels]
        crf_valid_labels = [' '.join(labels) for labels in valid_labels]
        crf_test_sents = [' '.join(sent) for sent in test_sents]
        crf_test_labels = [' '.join(labels) for labels in test_labels]

        crf_train_dict = {"sentences": crf_train_sents, "labels": crf_train_labels}  # sent in semeval instead of sentences
        crf_val_dict = {"sentences": crf_valid_sents, "labels": crf_valid_labels}
        crf_test_dict = {"sentences": crf_test_sents, "labels": crf_test_labels}
        # to store them to run crf from semeval 2023
        #crf_train_df = pd.DataFrame.from_dict(crf_train_dict)
        #crf_val_df = pd.DataFrame.from_dict(crf_val_dict)
        #crf_test_df = pd.DataFrame.from_dict(crf_test_dict)
        #crf_train_df.to_csv("distilbert_train.csv", index=False)
        #crf_val_df.to_csv("distilbert_val.csv", index=False)
        #crf_test_df.to_csv("distilbert_test.csv", index=False)

        crf_train_set = Dataset.from_dict(crf_train_dict)
        crf_val_set = Dataset.from_dict(crf_val_dict)
        #print(crf_train_set[4]["sentences"])
        #print(crf_train_set[4]["labels"])
        train_crf = CustomTrainer(model, crf_train_set, crf_val_set, labels_to_ids)
        train_crf.model_trainer(output_path)

    elif type_model == "distilbert":
        model = DistilbertNER(unique_labels)

        chosen_tokenizer = model.tokenizer

        train_dataset = AnnotatedDataset(train_labels, train_sents, labels_to_ids, tokenizer=chosen_tokenizer)
        valid_dataset = AnnotatedDataset(valid_labels, valid_sents, labels_to_ids, tokenizer=chosen_tokenizer)    
        test_dataset = AnnotatedDataset(test_labels, test_sents, labels_to_ids, tokenizer=chosen_tokenizer)    

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
        
        
        trainer = NERTrainer(model.pretrained, train=True)
        trained_model = trainer.epoch_loop(4, train_dataloader, valid_dataloader)  # seems to overfit after epoch 3

        print('\n'+"-"*10+"Token Evaluation"+"-"*10)
        evaluation = NEREvaluation(trained_model, train=False)
        evaluation.epoch_loop(1, test_dataloader)

        pred_sent_labels = [' '.join([ids_to_labels[pred] for pred in pred_sent]) for pred_sent in evaluation.pred_sents]
        true_sent_labels = [' '.join([ids_to_labels[true] for true in true_sent]) for true_sent in evaluation.label_sents]
        pred_tokens = [tok for sent in pred_sent_labels for tok in sent.split(' ')]
        true_tokens = [tok for sent in true_sent_labels for tok in sent.split(' ')]

        report = classification_report(pred_tokens, true_tokens)
        print(report)
    #storing_data = {"ground_truths_gold": true_sent_labels, "predictions": pred_sent_labels}
    #df = pd.DataFrame(storing_data)
    #df.to_csv("./bert_results.csv", header=True, index=False)

    print('\n'+"-"*10+"Span Evaluation"+"-"*10)
    #compute_metrics(pred_labels, true_labels, unique_labels)

if __name__ == "__main__":
    main(sys.argv[1:], "distilbert")
    #main(sys.argv[1:], "crf_distilbert")

