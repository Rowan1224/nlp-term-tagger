from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
import random
from torch.utils.data import DataLoader
import copy
import pandas as pd
from sklearn.metrics import classification_report
from entityDataset import EntityDataset
from model import RNN_CRF, RNN
from args import create_arg_parser
from train import train_step
from eval import eval
from utils import span_evaluation
import os

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def training_lstm(model, train_dataloader, valid_dataloader, num_epochs, learning_rate, device='cpu', verbose=True, use_crf=False):

    # Make a copy of the model (avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)
    
    
    # Set the model in 'training' mode (ensures all parameters' gradients are computed - it's like setting 'requires_grad=True' for all parameters)
    model_tr.train()
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)
    
    # Initialize lists to record the training loss over epochs
    loss_all_epochs = []
    val_loss_all_epochs = []
    
    best_accuracy = 0.0
    
    
    accuracy = []
    
    
    # Training loop
    for epoch in range(num_epochs):
        # Initialize the training loss for the current epoch
        loss_current_epoch = train_step(model_tr, train_dataloader, optimizer)
        val_loss_epoch = train_step(model_tr, valid_dataloader, optimizer, validation=True)


        acc = eval(model_tr, valid_dataloader,device=device)
        
        accuracy.append(acc)
        if acc > best_accuracy:
            best_accuracy = acc
            name = 'CRF' if use_crf else 'Base'
            if not os.path.exists('./output/'):
                os.mkdir('./output')

            torch.save(model_tr.state_dict(), f'output/model-{name}.pt')
            
        
        
        if verbose:
            print('Epoch [{}/{}],Train Loss: {:.4f} Val Loss: {:.4f}'.format(epoch+1, num_epochs, loss_current_epoch, val_loss_epoch))
        
    return model_tr, loss_all_epochs ,accuracy


def main():


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    args = create_arg_parser()

    VECTOR_PATH = args.vector_path
    EMB_DIMENSION = args.embedding_size
    MAX_SEQ_LENGTH = args.seq_length


    dataset_train = EntityDataset('../Dataset/train',VECTOR_PATH, EMB_DIMENSION,MAX_SEQ_LENGTH,  device)
    dataset_test = EntityDataset('../Dataset/test',VECTOR_PATH,EMB_DIMENSION, MAX_SEQ_LENGTH, device)
    dataset_dev = EntityDataset('../Dataset/dev',VECTOR_PATH,EMB_DIMENSION, MAX_SEQ_LENGTH, device)
    

    VOCAB = dataset_train.vocab


    batch_size = args.batch
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)


    #the vocab size that is built from train set
    vocab_size = len(dataset_train.vocab)
    # the embedding dimenstion 50/100/300
    emb_dim = EMB_DIMENSION
    # get the embedding matrix
    word_embeddings = dataset_train.word_embeddings
    # max sequence length
    max_sequence_length = MAX_SEQ_LENGTH

    #define lstm layers
    num_layers = args.layers
    #define hidden size
    hidden_size = args.hidden_size
    #set if LSTM should be bidirectional 
    bidirectional = True
    # output size i.e class size 
    output_size = 3
    # activation function
    act_fn = nn.LogSoftmax(dim=-1)

    use_crf = args.use_crf

    # create a RNN  model instance. REMARK: remove .cuda() at the end if gpu is not available
    
    if use_crf:
        rnn = RNN_CRF(vocab_size, emb_dim, word_embeddings, max_sequence_length, 
            num_layers,hidden_size, bidirectional, output_size, act_fn, device)
    else:
        rnn = RNN(vocab_size, emb_dim, word_embeddings, max_sequence_length, 
            num_layers,hidden_size, bidirectional, output_size, act_fn, device)

    # 


    rnn.to(device)

    # number of epochs
    num_epochs = args.epoch
    # learning rate
    learning_rate = args.learning_rate

    # train model
    model_tr, loss_all_epochs, accuracy = training_lstm(rnn, train_dataloader, valid_dataloader, num_epochs, learning_rate, device, use_crf=use_crf)

    # model_dir = './output/model-CRF.pt' if use_crf else './output/model-Base.pt'
    # state_dict = torch.load(model_dir)
    # rnn.load_state_dict(state_dict)
    # acc, preds = eval(rnn,test_dataloader,VOCAB, MAX_SEQ_LENGTH, device, True, use_crf)

    acc, preds = eval(model_tr,test_dataloader,VOCAB, MAX_SEQ_LENGTH, device, True, use_crf)
    outputs=[]
    sents = []
    pred_labels=[]
    true_labels = []
    true_spans = []
    pred_spans = []
    for o,p,t,s in preds:
        outputs.extend(o)
        sents.extend(s)
        
        for i in range(len(p)):
            pred_labels.extend(p[i])
            true_labels.extend(t[i])
            true_spans.append(" ".join(p[i]))
            pred_spans.append(" ".join(t[i]))


    print("#"*50)
    print('Token Level Evaluation')
    print(classification_report(pred_labels,true_labels))
    print("#"*50)

    df = pd.DataFrame()
    df['sents'] = sents
    df['true'] = true_spans
    df['preds'] = pred_spans


    print("#"*50)
    print('Span Level Evaluation')
    print("#"*50)
    span_evaluation(true_spans,pred_spans)
    print("#"*50)


    name = 'CRF' if use_crf else 'Base'
    df.to_csv(f'./output/outputs-{name}.csv',index=False)



if __name__ == '__main__':
    main()
