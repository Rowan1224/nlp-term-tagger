# sources
# https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
# https://pythonawesome.com/pytorch-named-entity-recognition-with-bert/ 
## with pyspark but y?
# https://sparkbyexamples.com/pyspark-tutorial/
# https://github.com/kamalkraj/BERT-NER
# https://towardsdatascience.com/custom-named-entity-recognition-with-bert-cf1fd4510804
from transformers import BertTokenizerFast, DistilBertTokenizer
from pipeline import AnnotatedDataset
from models import BertModel, DistilbertNER
import torch.optim as optim
import torch
import copy
from torch.utils.data import DataLoader, random_split, Subset
import sys
import tqdm


class NERTrainer:

    def __init__(self, model, loss=1000, acc=0, lr=2e-5, train=False):
        self.best_loss = loss  # unused
        self.best_acc = acc  # unused
        self.total_acc = 0
        self.total_loss = 0
        self.train = train
        #self.total_acc_val = 0
        #self.total_loss_val = 0
        if self.train:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model_tr = copy.deepcopy(model)
        # self.device = torch.device("cpu")
        # if we have gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
           self.model_tr = self.model_tr.cuda()

    def epoch_loop(self, epochs, train_data):


        for epoch in range(epochs):
            if self.train:
                self.model_tr.train()
            else:
                self.model_tr.eval()
            self.total_acc = 0
            self.total_loss = 0
            epoch_acc, epoch_loss = self.train_val_loop(train_data)
            print(f"Epoch {epoch:.3f} | Loss {epoch_loss:.3f} | Accuracy {epoch_acc:.3f}")

    def train_val_loop(self, train_data):  # may need to pass to DataSequence

        for data, label in train_data:
            label = label.to(self.device)
            mask = data["attention_mask"].squeeze(1).to(self.device)  # eliminating dims of size 1
            input_id = data['input_ids'].squeeze(1).to(self.device)
            if self.train:
                self.optimizer.zero_grad()
            loss, logits = self.model_tr(input_id, mask, label)
            
            self.clean_logits(logits, label, loss)
            loss.backward()
            if self.train:
                self.optimizer.step()

        epoch_acc = self.total_acc / len(train_data)
        epoch_loss = self.total_loss / len(train_data)
        return epoch_acc, epoch_loss

    def clean_logits(self, logits, label, loss):

        for i in range(logits.shape[0]):
            logits_clean = logits[i][label[i] != -100]
            label_clean = label[i][label[i] != -100]

            preds = logits_clean.argmax(dim=1)
            self.total_acc += (preds == label_clean).float().mean()
            self.total_loss += loss.item()


def create_raw_data(annotations_file):

    with open(annotations_file, 'r') as concat_file:
            sents_labels = [line.strip() for line in concat_file]
    
    sents_labels = [sent_label.split("\t") for sent_label in sents_labels]
    sents, labels = zip(*sents_labels)

    labels = [label.split() for label in labels]
    
    unique_labels = set([l.upper() for label in labels for l in label])  # forcing uppercases due to errors
    return unique_labels, sents, labels

    

def main(annotation_files, type_model):
    """
    opens file, tokenizes and finds unique labels before feeding data to main class
    """


    #annotation_files = annotation_files[1:]  # ignoring python script
    unique_labels, train_sents, train_labels = create_raw_data(annotation_files[0])  # algined_labels_760
    _, valid_sents, valid_labels = create_raw_data(annotation_files[1])
    models = {"bert": BertModel(unique_labels), "distilbert": DistilbertNER(unique_labels)}        
    model = models[type_model]  # warning message, needs to fine tune! 
    chosen_tokenizer = model.tokenizer

    train_dataset = AnnotatedDataset(train_labels, train_sents, unique_labels, tokenizer=chosen_tokenizer)
    valid_dataset = AnnotatedDataset(valid_labels, valid_sents, unique_labels, tokenizer=chosen_tokenizer)    


    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
    
    trainer = NERTrainer(model, True)
    trainer.epoch_loop(1, train_dataloader)
    validation = NERTrainer(model, False)
    validation.epoch_loop(1, valid_dataloader)

if __name__ == "__main__":
    main(sys.argv[1:], "bert")

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
