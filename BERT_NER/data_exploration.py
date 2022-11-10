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

    def __init__(self, df, unique_labels):
        self.df = df
        self.df["labels"] = self.df["labels"].apply(str.upper)  # fixing lowercase annotations...
        self.labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        self.ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
        self.align_labels()

    def __len__(self):
        return len(self.df["aligned_labels"])
        return self.df.size


    def align_labels(self):

        self.df["word_ids"] = [token_text.word_ids() for token_text in self.df["tokenized"]]
        word_ids = self.df["word_ids"].values.tolist()
        word_labels = self.df["labels"].apply(str.split).values.tolist()
        pre_alignment = list(zip(word_ids, word_labels))
    
        aligned_labels = []
        for word_idx, labels in pre_alignment:
            previous_idx = None
            label_ids = []
            for idx in word_idx:

                if idx is None:
                    label_ids.append(-100)
                
                elif idx != previous_idx:
                    try:
                        label_ids.append(self.labels_to_ids[labels[idx]])
                    except IndexError:
                        #print("Word ids may have idx beyond the length of the labels")
                        continue

                else:
                    label_ids.append(self.labels_to_ids[labels[idx]])
                previous_idx = idx
            aligned_labels.append(label_ids)
            #test = self.df["tokenized"].iloc[0].input_ids
            #print(self.tokenizer.convert_ids_to_tokens(test))
        self.df["aligned_labels"] = aligned_labels


def tokenizer(sents) -> BertTokenizerFast:
    """
    sents: list of str
    """

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    return [tokenizer(sent, padding="max_length", max_length=512, truncation=True,
              return_tensors="pt") for sent in sents]

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

    annotated_df = pd.DataFrame(sents_labels, columns = ["sentences", "labels"])
    annotated_df["tokenized"] = tokenizer(sents)
    annotations = AnnotatedDataset(annotated_df, unique_labels)



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
