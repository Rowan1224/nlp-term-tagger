import copy
import torch
import numpy as np

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
        # tags = [indexMap[idx] for idx in batch]
        
        batchTags.append(tags)
    
    return batchTags

def index_to_token(token_ids, VOCAB):
    """
    * For each item in the batch of token indices, convert the token indices to strings using the
    vocabulary
    
    :param token_ids: a batch of token indices
    :param VOCAB: the vocabulary of the dataset
    :return: A list of lists of strings.
    """
    
    """convert a batch of token indices to list of strings"""
    
    batchSent = []
    
    for item in token_ids:
    
        sent = [VOCAB[idx-1] if idx < len(VOCAB) else 'UNK' for idx in item if idx!=0]
        
        batchSent.append(sent)
    
    return batchSent


def print_predictions(tokens, pred_tags, true_tags, VOCAB, MAX_SEQ_LENGTH=96):
    """
    It takes in the model, the output of the model, the labels, the batch size, the max sequence length,
    and the device
    
    :param tokens: the tokenized text
    :param pred_tags: the predicted tags
    :param true_tags: the true tags for the batch
    :param VOCAB: The vocabulary of the dataset
    :param MAX_SEQ_LENGTH: The maximum length of a sentence, defaults to 96 (optional)
    """
    
    
    batch_tokens = index_to_token(tokens, VOCAB)
      
    batch_pred_tags = index_to_tag(pred_tags, MAX_SEQ_LENGTH)
    
    batch_true_tags = index_to_tag(true_tags, MAX_SEQ_LENGTH)
        
    
    from colorama import Style, Back
    
    outputs = []
    
    preds = []
    
    true = []
    
    sents = []

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
        sents.append(" ".join(tokens))
    
    return outputs, preds, true, sents


def eval_crf(model, outputs, labels, batch=8, max_sequence_length=96, device = 'cpu'):

    num_class = outputs.shape[-1]

    batch = int(np.prod(outputs.shape)/(max_sequence_length*num_class))


    outputs = outputs.reshape(batch,max_sequence_length,num_class)

    flat_labels = labels.reshape(-1,num_class)
    pad_index=[1 if flat_labels[i].sum()!=0 else 0 for i in range(flat_labels.shape[0])]
    mask = torch.FloatTensor(pad_index)
    mask = mask.to(device)
    mask = mask.reshape(batch,max_sequence_length)

    labels = torch.argmax(labels, dim=-1).to(device)
        
    
    best_path = model.crf_layer.viterbi_tags(outputs, mask)
    predictions = np.zeros(labels.shape)

    best_path = [ids for ids,_ in best_path]

    # true_labels = [labels[i][:len(best_path[i])].tolist() for i in range(labels.shape[0])]
    for i in range(predictions.shape[0]):
        predictions[i][:len(best_path[i])] = best_path[i]
        
    # print(predictions)

    return labels, torch.tensor(predictions).to(device)
    



def eval(model, eval_dataloader, VOCAB=None, MAX_SEQ_LENGTH=96, device='cpu', return_predictions = False, CRF=False):
    """
    The function takes in a model, a dataloader, a vocabulary, a maximum sequence length, and a device.
    It then sets the model to evaluation mode, and iterates over the dataloader. For each batch, it gets
    the sentences and labels, and gets the number of classes. It then finds the padded tokens, reshapes
    it to make it the same shape as the labels, counts the non-pad tokens, and counts the padded tokens.
    If the model is a CRF model, it gets the output, and then calls the eval_crf function to get the
    labels and label_predicted. If the model is not a CRF model, it gets the predicted labels, and then
    reshapes the labels to make it the same shape as the model output. It then computes the accuracy,
    and returns the accuracy.
    
    :param model: the model to be evaluated
    :param eval_dataloader: the dataloader for the evaluation dataset
    :param VOCAB: the vocabulary object
    :param MAX_SEQ_LENGTH: the maximum length of the input sequence, defaults to 96 (optional)
    :param device: the device on which the model is trained, defaults to cpu (optional)
    :param return_predictions: if True, the function will return the predictions as well as the
    accuracy, defaults to False (optional)
    :param CRF: whether to use CRF or not, defaults to False (optional)
    """
    
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
            

            
            if CRF:
                out = model(sent)
                labels, label_predicted = eval_crf(model, out, labels, len(batch), MAX_SEQ_LENGTH,device)
                # predictions.append(print_predictions(sent,label_predicted,labels, VOCAB, MAX_SEQ_LENGTH))


            else:

                # Get the predicted labels
                y_predicted = model(sent)
                # To get the predicted labels, we need to get the max over all possible classes
                # multiply with padx to ignore padded token predictions 
                label_predicted = torch.argmax(y_predicted.data, 1)*padx

                #reshape it to make it as the same shape with model output
                labels = labels.reshape(-1,num_class)
                labels = torch.argmax(labels, 1)*padx
                    
                # print(labels)

            # Compute accuracy: count the total number of samples,
            #and the correct labels (compare the true and predicted labels)
                
            total_labels += num_tokens #only added the non-padded tokens in count
            
            # subtract the padded tokens to ignore padded token predictions in final count
            correct_labels += ((label_predicted == labels).sum().item() - num_pad_tokens)
        
            accuracy = 100 * correct_labels / total_labels
            # get output
            if return_predictions:
                predictions.append(print_predictions(sent,label_predicted,labels, VOCAB, MAX_SEQ_LENGTH))
                
            
        if return_predictions:
            return 0, predictions
        
        return accuracy

