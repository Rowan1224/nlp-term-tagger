from transformers import BertForTokenClassification, DistilBertForTokenClassification
import torch.nn as nn

class BertModel(nn.Module):

    def __init__(self, unique_labels):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output


class DistilbertNER(nn.Module):
  """
  Implement NN class based on distilbert pretrained from Hugging face.
  Inputs :
    tokens_dim : int specifyng the dimension of the classifier
  """

  def __init__(self, unique_labels):
    super(DistilbertNER,self).__init__()

    #if type(tokens_dim) != int:
    #        raise TypeError('Please tokens_dim should be an integer')

    #if tokens_dim <= 0:
    #      raise ValueError('Classification layer dimension should be at least 1')

    self.pretrained = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(unique_labels)) #set the output of each token classifier = unique_lables


  def forward(self, input_ids, attention_mask, labels=None): #labels are needed in order to compute the loss
    """
  Forwad computation of the network
  Input:
    - inputs_ids : from model tokenizer
    - attention :  mask from model tokenizer
    - labels : if given the model is able to return the loss value
  """

    #inference time no labels
    if labels == None:
      out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
      return out

    out = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
    return out
