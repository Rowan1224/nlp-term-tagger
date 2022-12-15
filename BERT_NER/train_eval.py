import torch.optim as optim
import torch
import copy
import tqdm


class NERTrainer:

    def __init__(self, model, lr=1e-5, train=True):

        self.total_acc = 0
        self.total_loss = 0
        self.train = train
        self.model = copy.deepcopy(model)
        if self.train:
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

        # self.device = torch.device("cpu")
        # if we have gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = 42
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)


    def epoch_loop(self, epochs, train_data, val_data):

        self.model.to(self.device)

        if self.train:
            self.model.train()

        for epoch in range(epochs):
 
            self.train = True
            epoch_acc, epoch_loss = self.train_val_loop(train_data)
            print(f"Train Epoch {epoch+1} | Loss {epoch_loss:.3f} | Accuracy {epoch_acc:.3f}")            

            self.train = False
            self.model.eval()
            epoch_acc, epoch_loss = self.train_val_loop(val_data)
            print(f"Val Epoch {epoch+1} | Loss {epoch_loss:.3f} | Accuracy {epoch_acc:.3f}")

        return self.model

    def train_val_loop(self, train_data):  # may need to pass to DataSequence
        self.total_acc = 0
        self.total_loss = 0

        for data, label in train_data:
            
            label = label.to(self.device)
            mask = data["attention_mask"].to(self.device) 
            input_id = data['input_ids'].to(self.device)

            if self.train:
                self.optimizer.zero_grad()
            loss, logits = self.model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

            if self.train:
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                loss.backward()
                self.optimizer.step()

            self.clean_logits(logits, label, loss)
            
        epoch_acc = self.total_acc / len(train_data)
        epoch_loss = self.total_loss / len(train_data)
        return epoch_acc, epoch_loss

    def clean_logits(self, logits, label, loss):

        batch_size = logits.shape[0]
        for i in range(batch_size):
            
            mask = label[i] != -100

            label_clean = torch.masked_select(label[i], mask)

            preds = torch.masked_select(logits[i].argmax(dim=1), mask)

            self.total_acc += (preds == label_clean).float().mean()/batch_size
            #self.total_acc += (preds == label_clean).float().mean()

        self.total_loss += loss.item()


class NEREvaluation(NERTrainer):

    def __init__(self, model):
        self.total_labels = 0
        self.correct_labels = 0
        self.pred_sents = []
        self.label_sents = []
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluating(self, eval_data):

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            batch_acc = self.eval_loop(eval_data)
        print(f"Accuracy {batch_acc:.3f}%")

        return batch_acc
    
    def eval_loop(self, eval_data):  # may need to pass to DataSequence

        for data, label in eval_data:
            
            label = label.to(self.device)
            mask = data["attention_mask"].to(self.device) 
            input_id = data['input_ids'].to(self.device)

            _, logits = self.model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)  # loss

            self.clean_logits(logits, label)
            
        eval_acc = self.correct_labels / self.total_labels  #  - num of padded_tokens? change correct from mean to sum
        return eval_acc * 100

    def clean_logits(self, logits, label):

        batch_size = logits.shape[0]
        for i in range(batch_size):
            mask = label[i] != -100
            label_clean = torch.masked_select(label[i], mask)
            self.total_labels += label_clean.shape[0]
            preds = torch.masked_select(logits[i].argmax(dim=1), mask)
            self.correct_labels += (preds == label_clean).sum().item()
            #self.correct_labels += (preds == label_clean).float().mean()
            self.pred_sents.append(preds.cpu().numpy())
            self.label_sents.append(label_clean.cpu().numpy())


def create_raw_data(annotations_file):

    with open(annotations_file, 'r') as concat_file:
            sents_labels = [line.strip() for line in concat_file]
    
    sents_labels = [sent_label.split("\t") for sent_label in sents_labels]
    sents, labels = zip(*sents_labels)
    sents = [sent.split() for sent in sents]           

    labels = [label.split() for label in labels]
    
    unique_labels = set([l.upper() for label in labels for l in label])  # forcing uppercases due to errors
    return unique_labels, sents, labels
