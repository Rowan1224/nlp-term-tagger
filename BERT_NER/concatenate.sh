#!/bin/bash

#paste ../Dataset/train/sentences.txt ../Dataset/train/labels.txt
#paste $1 $2 > entities_labels.txt
#python <(paste $1 $2)
#bash ../Dataset/train/sentences.txt ../Dataset/train/labels.txt  HOW IT'S RUN
# define variables with paths instead!
train_sents="../Dataset/train/sentences.txt"
train_labels="../Dataset/train/labels.txt"
valid_sents="../Dataset/dev/sentences.txt"
valid_labels="../Dataset/dev/labels.txt"
test_sents="../Dataset/test/sentences.txt"
test_labels="../Dataset/test/labels.txt"
#echo $train_sents
#python data_exploration.py <(paste $1 $2)
python train.py <(paste $train_sents $train_labels) <(paste $valid_sents $valid_labels)
