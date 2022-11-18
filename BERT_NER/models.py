from transformers import BertForTokenClassification, DistilBertForTokenClassification, BertTokenizerFast, DistilBertTokenizerFast
import torch.nn as nn

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



