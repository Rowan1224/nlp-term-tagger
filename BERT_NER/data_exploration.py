# sources
# https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
# https://pythonawesome.com/pytorch-named-entity-recognition-with-bert/ 
## with pyspark but y?
# https://sparkbyexamples.com/pyspark-tutorial/
# https://github.com/kamalkraj/BERT-NER
# https://towardsdatascience.com/custom-named-entity-recognition-with-bert-cf1fd4510804
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizerFast
from transformers import pipeline
from torch.utils.data import Dataset
import pandas as pd
import sys

class AnnotatedDataset(Dataset):

    def __init__(self, tokenizer, df, unique_labels):
        self.df = df
        self.df["labels"] = self.df["labels"].apply(str.upper)  # fixing lowercase annotations...
        self.labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        self.ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
        self.tokenizer = tokenizer

    def tokenize(self):
        self.df["tokenized"] = self.df["sentences"].apply(self.tokenizer)
        self.df["word_ids"] = [token_text.word_ids() for token_text in self.df["tokenized"]]
        #self.df[["word_ids", "labels"]].apply(self.align_label_example)
        self.df[["word_ids", "labels"]]
        self.df.apply(self.align_label_example(self.df["word_ids"], self.df["labels"]), axis=1)


#TODO: Fix alignment
    #def align_label_example(tokenized_input, labels):
    def align_label_example(self, word_ids, labels):
    

            # TODO: labels are self.df["labels"]
            #print(row["word_ids"])
    
            previous_word_idx = None
            label_ids = []
            print(word_ids)
            print(labels)
            raise SystemExit
    
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)


                
                elif word_idx != previous_word_idx:
                    #print(word_idx)  # returns whole array...
                    #print(previous_word_idx)
                    #x = self.df["labels"][word_idx]  # ERROR
                    #print(x)
                    #break
                    continue

                else:
                    pass

                previous_word_idx = word_idx
    
            #    elif word_idx != previous_word_idx:
            #        try:
            #          label_ids.append(labels_to_ids[labels[word_idx]])
            #        except:
            #          label_ids.append(-100)
    
            #    else:
            #        label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            #        label_ids.append(labels_to_ids[labels[word_idx]])
            #    previous_word_idx = word_idx
    
    
            #return label_ids


def main(annotations_file):

    with open(annotations_file, 'r') as concat_file:
            sents_labels = [line.strip() for line in concat_file]
    
    sents_labels = [sent_label.split("\t") for sent_label in sents_labels]
    labels = [sent_label[1].split() for sent_label in sents_labels]
    unique_labels = set([l.upper() for label in labels for l in label])  # forcing uppercases due to errors
    annotated_df = pd.DataFrame(sents_labels, columns = ["sentences", "labels"])
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased", padding="max_length",
                                                  max_length=512, truncation=True,
                                                  return_tensors="pt")
    annotations = AnnotatedDataset(tokenizer, annotated_df, unique_labels)
    annotations.tokenize()

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
