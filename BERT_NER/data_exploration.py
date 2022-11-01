# sources
# https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
# https://pythonawesome.com/pytorch-named-entity-recognition-with-bert/ 
## with pyspark but y?
# https://sparkbyexamples.com/pyspark-tutorial/
import pandas as pd
import sys


annotations_path = "../Dataset/train"
sentences = annotations_path + "/sentences.txt"

with open(sys.argv[1], 'r') as concat_file:
        sents_labels = [line.strip() for line in concat_file]

sents_labels = [sent_label.split("\t") for sent_label in sents_labels]
print(sents_labels[:3])
df = pd.DataFrame(sents_labels, columns = ["sentences", "labels"])


prin(df.head())
# source for pipes https://stackoverflow.com/questions/11109859/pipe-output-from-shell-command-to-a-python-script
