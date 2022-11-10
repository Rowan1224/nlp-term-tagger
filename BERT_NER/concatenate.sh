#!/bin/bash

#paste ../Dataset/train/sentences.txt ../Dataset/train/labels.txt
#paste $1 $2 > entities_labels.txt
#python <(paste $1 $2)
#bash ../Dataset/train/sentences.txt ../Dataset/train/labels.txt  HOW IT'S RUN
# define variables with paths instead!
python data_exploration.py <(paste $1 $2)
