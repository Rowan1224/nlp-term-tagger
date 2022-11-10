# sources
# https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
# https://pythonawesome.com/pytorch-named-entity-recognition-with-bert/ 
## with pyspark but y?
# https://sparkbyexamples.com/pyspark-tutorial/
# https://github.com/kamalkraj/BERT-NER
# https://towardsdatascience.com/custom-named-entity-recognition-with-bert-cf1fd4510804
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizerFast
from pipeline import AnnotatedDataset
from models import BertModel
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, random_split, Subset
import sys

def tokenizer(sents) -> BertTokenizerFast:
    """
    sents: list of str
    """

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    return [tokenizer(sent, padding="max_length", max_length=512, truncation=True,
              return_tensors="pt") for sent in sents]

def train(model, train_data, val_data):
    """

    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)

def main(annotations_file):
    """
    opens file, tokenizes and finds unique labels before feeding data to main class
    """

    with open(annotations_file, 'r') as concat_file:
            sents_labels = [line.strip() for line in concat_file]
    
    sents_labels = [sent_label.split("\t") for sent_label in sents_labels]
    sents, labels = zip(*sents_labels)

    labels = [label.split() for label in labels]
    unique_labels = set([l.upper() for label in labels for l in label])  # forcing uppercases due to errors

    tokenized = tokenizer(sents)
    annotations = AnnotatedDataset(labels, tokenized, unique_labels)
    train_dataloader = DataLoader(annotations, batch_size=8, shuffle=False)
    #val_dataloader = DataLoader(annotations, batch_size=8, shuffle=False)

    BertModel(unique_labels)

if __name__ == "__main__":
    main(sys.argv[1])

# BERT Example
# https://huggingface.co/dslim/bert-base-NER?text=My+name+is+Wolfgang+and+I+live+in+Berlin
#tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
#model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
#
#nlp = pipeline("ner", model=model, tokenizer=tokenizer)
#example = "My name is Wolfgang and I live in Berlin"
#
#ner_results = nlp(example)
#print(ner_results)
# source for pipes https://stackoverflow.com/questions/11109859/pipe-output-from-shell-command-to-a-python-script
