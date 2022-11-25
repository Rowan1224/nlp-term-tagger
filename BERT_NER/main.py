from alignment import AnnotatedDataset
from models import BertModel, DistilbertNER
from train_eval import NERTrainer, NEREvaluation, create_raw_data
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import classification_report
import tqdm


def main(annotation_files, type_model):
    """
    opens file, tokenizes and finds unique labels before feeding data to main class
    """


    #annotation_files = annotation_files[1:]  # ignoring python script
    unique_labels, train_sents, train_labels = create_raw_data(annotation_files[0])  # algined_labels_760
    _, valid_sents, valid_labels = create_raw_data(annotation_files[1])  # ignore unique labels from val, they are both the same
    _, test_sents, test_labels = create_raw_data(annotation_files[2])  # ignore unique labels from val, they are both the same
    if type_model == "bert":
        model = BertModel(unique_labels)
    else:
        model = DistilbertNER(unique_labels)
    #models = {"bert": BertModel(unique_labels), "distilbert": DistilbertNER(unique_labels)}        
    #model = models[type_model]  # warning message, needs to fine tune! 
    chosen_tokenizer = model.tokenizer

    train_dataset = AnnotatedDataset(train_labels, train_sents, unique_labels, tokenizer=chosen_tokenizer)
    valid_dataset = AnnotatedDataset(valid_labels, valid_sents, unique_labels, tokenizer=chosen_tokenizer)    
    test_dataset = AnnotatedDataset(test_labels, test_sents, unique_labels, tokenizer=chosen_tokenizer)    


    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    
    
    trainer = NERTrainer(model.pretrained, train=True)
    trained_model = trainer.epoch_loop(4, train_dataloader, valid_dataloader)  # seems to overfit after epoch 3

    print('\n'+"-"*10+"Evaluation"+"-"*10)
    evaluation = NEREvaluation(trained_model, train=False)
    evaluation.epoch_loop(1, test_dataloader)
    pred_labels = evaluation.preds_viz
    true_labels = evaluation.labels_viz
    report = classification_report(pred_labels, true_labels)
    print(report)
    # span based analysis

if __name__ == "__main__":
    main(sys.argv[1:], "distilbert")
