#!/bin/sh

#echo Running $1 with $2 configuration

if [ "$1" = "lstm" ]; then
    cd ./LSTM
    if [ "$2" = "crf" ]; then
        python lstm.py -c #-b 8 -epoch 10 -emb 100 -l 128 -layers 5 -hidden 32 -lr 0.01 -v ./glove.6B/glove.6B.100d.txt 
    else
        python lstm.py #-b 8 -epoch 10 -emb 100 -l 128 -layers 5 -hidden 32 -lr 0.01 -v ./glove.6B/glove.6B.100d.txt
    fi
elif [ "$1" = "bert" ]; then
    cd ./PLM
    if [ "$2" = "crf" ]; then
        python train.py -c #-b 4 -epoch 10 -l 128 -lr 1e-5
        pythom eval.py -c #-b 4 -epoch 10 -l 128 -lr 1e-5
    else
        python train.py #-b 4 -epoch 10 -l 128 -lr 1e-5 -m bert
        pythom eval.py #-b 4 -epoch 10 -l 128 -lr 1e-5 -m bert
    fi
else
    cd ./Baseline
    python baseline.py
fi
