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
import copy
from torch.utils.data import DataLoader, random_split, Subset
import sys
import tqdm

def tokenizer(sents) -> BertTokenizerFast:
    """
    sents: list of str
    """

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    return [tokenizer(sent, padding="max_length", max_length=512, truncation=True,
            return_tensors="pt") for sent in sents]

class NERTrainer:

    def __init__(self, loss=1000, acc=0, lr=0.01, model):
        self.best_loss = loss
        self.best_acc = acc
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        self.model_tr = model

    def trainer_loop(

def train(model, num_epochs, train_data, val_data):
    """

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tr = copy.deepcopy(model)
    if torch.cuda.is_available():
        model_tr = model_tr.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    best_acc = 0
    best_loss = 1000

    for epoch in range(num_epochs):  # change to tqdm
        model_tr.train()

        acc_train = 0
        loss_train = 0
        acc_val = 0
        loss_val = 0

        #for train_data, train_label in tqdm(train_data):
        train_val_loop(model_tr, train_data, acc_train, loss_train, optimizer, train=True)
        model_tr.eval()
        train_val_loop(model_tr, val_data, acc_val, loss_val, optimizer)  # we don't need opt fix logic
def train_val_loop(mode, train_data, acc, loss, optimizer, train=False):
    
    for data, label in train_data:  # can change to tqdm

        label = label.to(device)
        mask = data["attention_mask"].squeeze(1).to(device)  # eliminating dims of size 1
        input_id = data['input_ids'].squeeze(1).to(device)
        if train:
            optimizer.zero_grad()  # figure out what this is doing
        loss, logits = model(input_id, mask, label)
        clean_logits(logits, label, acc, loss)  # we need the acc value from here
        loss.backward()
        optimizer.step()
    accuracy = acc / len(train_data)
    loss = loss / len(train_data)
    return accuracy, loss

def clean_logits(logits, label, acc, loss):
    # may need to change repeated words labels to -100...

    for i in range(logits.shape[0]):

        logits_clean = logits[i][label[i] != -100]
        label_clean = label[i][label[i] != -100]

        preds = logits_clean.argmax(dim=1)
        acc += (predictions == label_clean).float().mean()
        loss += loss.item()

def train_loop(model, df_train, df_val):
    #outdated

    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][val_label[i] != -100]
              label_clean = val_label[i][val_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_val += acc
              total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

def create_raw_data(annotations_file):

    with open(annotations_file, 'r') as concat_file:
            sents_labels = [line.strip() for line in concat_file]
    
    sents_labels = [sent_label.split("\t") for sent_label in sents_labels]
    sents, labels = zip(*sents_labels)

    labels = [label.split() for label in labels]
    global unique_labels
    unique_labels = set([l.upper() for label in labels for l in label])  # forcing uppercases due to errors

    tokenized = tokenizer(sents)
    return AnnotatedDataset(labels, tokenized, unique_labels)

def main(annotation_files):
    """
    opens file, tokenizes and finds unique labels before feeding data to main class
    """


    #annotation_files = annotation_files[1:]  # ignoring python script
    train_dataset = create_raw_data(annotation_files[0])
    valid_dataset = create_raw_data(annotation_files[1])
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    model = BertModel(unique_labels)  # warning message, needs to fine tune!
    train(model, 5, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main(sys.argv[1:])

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
