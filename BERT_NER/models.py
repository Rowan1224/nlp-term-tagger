from transformers import BertForTokenClassification, DistilBertForTokenClassification, BertTokenizerFast, DistilBertTokenizerFast, DistilBertModel
#from allennlp.modules import ConditionalRandomField
#from allennlp.modules.conditional_random_field import allowed_transitions
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy


class BertModel():

    def __init__(self, unique_labels):
        self.bert = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(unique_labels))
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")




class DistilbertNER():
    """
    Implement NN class based on distilbert pretrained from Hugging face.
    Inputs :
      tokens_dim : int specifyng the dimension of the classifier
    """
    def __init__(self, unique_labels):
        self.pretrained = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(unique_labels)) #set the output of each token classifier = unique_lables
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
  

#class CRFDistilBERT(nn.Module):
#
#    # https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb
#    def __init__(self, ids_to_labels, num_unique_labels, device="cpu", dropout_rate=0.1):
#        super(CRFDistilBERT, self).__init__()
#        #self.pretrained_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased",
#        #                                                          max_position_embeddings=128, )  # change during training?
#        output_size = num_unique_labels
#        #ids_to_labels = {0: 'B', 1: 'I', 2: 'O'}
#        self.pretrained_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased") 
#        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased") 
#        
#        # output layer
#        self.feedforward = nn.Linear(in_features=self.pretrained_encoder.config.hidden_size, out_features=output_size)
#        self.crf_layer = ConditionalRandomField(num_tags=output_size, constraints=allowed_transitions(constraint_type="BIO", labels=ids_to_labels))
#        self.dropout = nn.Dropout(dropout_rate)
#        self.device = device
#      
#    def update_mask(self, labels, attention_mask):
#        shape = labels.size()
#        labelsU = torch.flatten(labels)
#        attention_maskU = torch.flatten(attention_mask)
#        idx = (labelsU == -100).nonzero(as_tuple=False)
#        idx = torch.flatten(idx)
#        labelsU[idx] = torch.tensor(0)
#        attention_maskU[idx] = torch.tensor(0)
#        labelsU = labelsU.reshape(shape)
#        attention_maskU = attention_maskU.reshape(shape)
#        return labelsU, attention_maskU
#
#
#    def forward(self, input_ids, attention_mask, labels):
#
#        batch_size = input_ids.size(0)
#        embedded_text_input = self.pretrained_encoder(input_ids=input_ids, attention_mask=attention_mask)
#        embedded_text_input = embedded_text_input.last_hidden_state
#
#        # embedded_text_input = self.dropout(F.leaky_rule(embedded_text_input))
#        token_scores = self.feedforward(embedded_text_input)
#
#        token_scores = F.log_softmax(token_scores, dim=-1)
#
#        labels = copy.deepcopy(labels)
#        attention_mask = copy.deepcopy(attention_mask)
#        labelsU, attention_maskU = self.update_mask(labels, attention_mask)
#        loss = self.crf_layer(token_scores, labelsU, attention_maskU) / float(batch_size)
#        best_path = self.crf_layer.viterbi_tags(token_scores, attention_maskU)
#        preds = torch.full(labels.size(), -100)
#
#        for i in range(batch_size):
#            idx, _ = best_path[i]
#            preds[i][:len(idx)] = torch.tensor(idx)
#
#        return {"loss": loss, "logits": preds.to(self.device)}



#crf_test = CRFDistilBERT(3)
#print(crf_test.pretrained_encoder)
