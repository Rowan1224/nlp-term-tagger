
from torch.utils.data import DataLoader
import torch
import numpy as np
import torch.nn as nn
import random
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from entityDataset import EntityDataset
from model import RNN
import argparse


def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", default=8, type=int, help="Provide the number of batch"
    )
    parser.add_argument(
        "-epoch", "--epoch", default=10, type=int, help="Provide the number of epochs"
    )
    parser.add_argument(
        "-layers", "--layers", default=5, type=int, help="Provide the number of batch"
    )
    parser.add_argument(
        "-hidden", "--hidden_size", default=32, type=int, help="Provide the number of epochs"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.01,
        type=float,
        help="Provide the learning rate",
    )
    parser.add_argument(
        "-l",
        "--seq_length",
        default=96,
        type=int,
        help="define Max sequence length",
    )

    parser.add_argument(
        "-e",
        "--embedding_size",
        default=100,
        type=int,
        choices=[50, 100, 300],
        help="Select the model type for training (fine-tuning or domain adaption on SQuAD model)",
    )

    parser.add_argument(
        "-v",
        "--vector_path",
        type=str,
        default="./glove.6B/glove.6B.100d.txt",
        help="Word Embedding Path",
    )

    args = parser.parse_args()
    return args


def index_to_tag(labels, MAX_SEQ_LENGTH=96):
    
    """convert a batch of label indices to list of tags"""
    
    #define index to tag mapping
    indexMap = {0:'B', 1:'I', 2:'O'}
    
    #reshape labels to batch_size*MAX_SEQ_LENGTH
    labels = labels.reshape((-1,MAX_SEQ_LENGTH))
    
    batchTags = []
    
    #convert label index to tags
    for batch in labels:
    
        tags = [indexMap[idx.item()] for idx in batch]
        
        batchTags.append(tags)
    
    return batchTags

def index_to_token(token_ids, VOCAB):
    
    """convert a batch of token indices to list of strings"""
    
    batchSent = []
    
    for item in token_ids:
    
        sent = [VOCAB[idx-1] if idx < len(VOCAB) else 'UNK' for idx in item if idx!=0]
        
        batchSent.append(sent)
    
    return batchSent


def print_predictions(tokens, pred_tags, true_tags, VOCAB, MAX_SEQ_LENGTH=96):
    
    
    batch_tokens = index_to_token(tokens, VOCAB)
      
    batch_pred_tags = index_to_tag(pred_tags, MAX_SEQ_LENGTH)
    
    batch_true_tags = index_to_tag(true_tags, MAX_SEQ_LENGTH)
        
    
    from colorama import Style, Back
    
    outputs = []
    
    preds = []
    
    true = []
    
    for tokens,true_tags,pred_tags in zip(batch_tokens,batch_pred_tags,batch_true_tags):
        
        true_tags = true_tags[:len(tokens)]
        pred_tags = pred_tags[:len(tokens)]
        
        output = []
    
        for t,tl,pl in zip(tokens,true_tags,pred_tags):

            assert len(tokens) == len(pred_tags) == len(true_tags)

            if tl == pl:
                o = f"{t} {Back.GREEN}[{tl}][{pl}]{Style.RESET_ALL}"

            else:
                o = f"{t} {Back.GREEN}[{tl}]{Style.RESET_ALL}{Back.RED}[{pl}]{Style.RESET_ALL}"


            output.append(o)
            
        outputs.append(" ".join(output))
        preds.append(pred_tags)
        true.append(true_tags)
    
    return outputs, preds, true



def eval_lstm(model, eval_dataloader, VOCAB=None, MAX_SEQ_LENGTH=96, return_predictions = False):
    
    model = copy.deepcopy(model)
    # Set the model in 'evaluation' mode (this disables some layers (batch norm, dropout...) which are not needed when testing)
    model.eval() 
    
    predictions = []

    # In evaluation phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        # initialize the total and correct number of labels to compute the accuracy
        correct_labels = 0
        total_labels = 0
        
        # Iterate over the dataset using the dataloader
        for batch in eval_dataloader:

            #get sentences and labels
            sent = batch['token_ids']
            labels = batch['labels']
            
            
            #get number of class or tags
            num_class = labels.shape[-1]
    
            #find the padded tokens
            padx = (sent > 0).float()
            
            #reshape it to make it as the same shape with labels
            padx = padx.reshape(-1)
  
            #count non-pad tokens
            num_tokens = padx.sum().item()
        
            #count padded tokens
            num_pad_tokens = padx.shape[0] - num_tokens
            
            #reshape it to make it as the same shape with model output
            labels = labels.reshape(-1,num_class)
            
            # Get the predicted labels
            y_predicted = model(sent)
            
            # To get the predicted labels, we need to get the max over all possible classes
            # multiply with padx to ignore padded token predictions 
            label_predicted = torch.argmax(y_predicted.data, 1)*padx
            labels = torch.argmax(labels, 1)*padx
            

            # Compute accuracy: count the total number of samples,
            #and the correct labels (compare the true and predicted labels)
            
            total_labels += num_tokens #only added the non-padded tokens in count
            
            # subtract the padded tokens to ignore padded token predictions in final count
            correct_labels += ((label_predicted == labels).sum().item() - num_pad_tokens)
            
            # get output
            if return_predictions:
                predictions.append(print_predictions(sent,label_predicted,labels, VOCAB, MAX_SEQ_LENGTH))
    
    accuracy = 100 * correct_labels / total_labels
    
    if return_predictions:
        return accuracy, predictions
    
    return accuracy


def loss_fn(outputs, labels):
    
    #define cross entropy loss 
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    #reshape labels to give a flat vector of length batch_size*seq_len
    num_class = labels.shape[-1]
    
    # reshape label to make it similar to model output
    labels = labels.reshape(-1,num_class) 

    #get loss
    loss = criterion(outputs, labels.float())
    
    #get non-pad index
    non_pad_index=[i for i in range(labels.shape[0]) if labels[i].sum()!=0]
    
    #get final loss
    loss = loss[non_pad_index].mean()
    
    return loss
    

      
    

def training_lstm(model, train_dataloader, valid_dataloader, num_epochs, learning_rate, verbose=True):

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
        loss_current_epoch = 0
        val_loss_epoch = 0
        
        # Iterate over batches using the dataloader
        for batch_index, batch in enumerate(train_dataloader):
            
            label = batch['labels']

            optimizer.zero_grad()
            
            out = model_tr.forward(batch['token_ids'])
            l = loss_fn(out,label)
            l.backward()
            optimizer.step()
            loss_current_epoch += (l.item())
            
            val_loss_epoch += loss_fn(out,label).item()


        # At the end of each epoch, record and display the loss over all batches in train and val set
        loss_current_epoch = loss_current_epoch/len(train_dataloader)
        val_loss_epoch = val_loss_epoch/len(train_dataloader)
        
        loss_all_epochs.append(loss_current_epoch)
        val_loss_all_epochs.append(val_loss_epoch)
        
        # 
        acc = eval_lstm(model_tr, valid_dataloader)
        
        accuracy.append(acc)
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model_tr.state_dict(), 'model_opt.pt')
            
        
        
        if verbose:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss_current_epoch))
        
    return model_tr, loss_all_epochs ,accuracy


def main():


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    args = create_arg_parser()

    VECTOR_PATH = args.vector_path
    EMB_DIMENSION = args.embedding_size
    MAX_SEQ_LENGTH = args.seq_length

    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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
    bidirectional = False
    # output size i.e class size 
    output_size = 3
    # activation function
    act_fn = nn.LogSoftmax(dim=1)

    # create a RNN  model instance. REMARK: remove .cuda() at the end if gpu is not available
    
    rnn = RNN(vocab_size, emb_dim, word_embeddings, max_sequence_length, 
            num_layers,hidden_size, bidirectional, output_size, act_fn, device)


    rnn.to(device)

    # number of epochs
    num_epochs = args.epoch
    # learning rate
    learning_rate = args.learning_rate

    # train model
    model_tr, loss_all_epochs, accuracy = training_lstm(rnn, train_dataloader, valid_dataloader, num_epochs, learning_rate)



    # plt.figure()
    # epochs = [i for i in range(num_epochs)]
    # plt.plot(epochs, loss_all_epochs, 'r', label='Loss')
    # plt.xlabel('epochs'), plt.ylabel('loss')
    # plt.legend()
    # plt.show()



    acc, preds = eval_lstm(model_tr,test_dataloader,VOCAB, MAX_SEQ_LENGTH, True)
    outputs=[]
    pred_labels=[]
    true_labels = []
    true_spans = []
    pred_spans = []
    for o,p,t in preds:
        outputs.extend(o)
        
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
    df['true'] = true_spans
    df['preds'] = pred_spans

    df.to_csv('outputs.csv',index=False)

    # for i, out in enumerate(outputs[:3]):
    #     print(out)
    #     print('\n')





if __name__ == '__main__':
    main()