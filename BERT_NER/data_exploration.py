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

    def __init__(self, tokenizer):
        tokenizer


#TODO: Fix alignment
def align_label_example(tokenized_input, labels):

        word_ids = tokenized_input.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                try:
                  label_ids.append(labels_to_ids[labels[word_idx]])
                except:
                  label_ids.append(-100)

            else:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx


        return label_ids


def main(annotations_file):

    with open(annotations_file, 'r') as concat_file:
            sents_labels = [line.strip() for line in concat_file]
    
    sents_labels = [sent_label.split("\t") for sent_label in sents_labels]
    annotated_df = pd.DataFrame(sents_labels, columns = ["sentences", "labels"])


    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    AnnotatedDataset(tokenizer)


examples = annotations_df["sentences"].values.tolist()
print(examples[5])

text_tokenized = tokenizer(examples[5], padding='max_length', max_length=512, truncation=True, return_tensors="pt")
res = tokenizer.decode(text_tokenized.input_ids[0])
print(res)
print(tokenizer.convert_ids_to_tokens(text_tokenized["input_ids"][0]))

word_ids = text_tokenized.word_ids()
print(tokenizer.convert_ids_to_tokens(text_tokenized["input_ids"][0]))
print(word_ids)

if __name__ = "__main__":
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
