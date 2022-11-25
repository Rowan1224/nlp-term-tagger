
import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, word_embeddings,
                 max_sequence_length, num_layers, hidden_size, bidirectional, output_size, act_fn, device ='cpu'):
        super(RNN, self).__init__()
        
        # embedding layer: converts tokens ids with respectve word vec
        self.input_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.input_layer.weight.data = word_embeddings
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size = hidden_size, 
                           num_layers = num_layers, 
                           bidirectional=bidirectional, batch_first=True)
        
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1
            
        self.layers = num_layers
        
        
            
        self.hidden_size = hidden_size

        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.direction*hidden_size, output_size),act_fn)

        self.device = device
    
    def forward(self, x):
        
        # get embedding 
        emb = self.input_layer(x)
        
        batch = x.shape[0]
        # initialize a hidden state and cell state
        h0,c0 = self.init_hidden(batch)
        
        # get output from lstm layers
        l,_ = self.lstm(emb.float(),(h0,c0))
        
        # flatten the output
        l = l.reshape(-1,l.shape[2])
    
        # get final class probabilities
        out = self.output_layer(l)
        
        return out
    
    def init_hidden(self, batch_size):
                
        torch.manual_seed(0)
        h0 = torch.randn(self.direction*self.layers, batch_size, self.hidden_size, device=self.device) 
        c0 = torch.randn(self.direction*self.layers, batch_size, self.hidden_size, device=self.device)

        return h0,c0