
import torch.nn as nn
import torch
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

class RNN_CRF(nn.Module):
    def __init__(self, vocab_size, emb_dim, word_embeddings,
                 max_sequence_length, num_layers, hidden_size, bidirectional, output_size, act_fn, device ='cpu'):
        super(RNN_CRF, self).__init__()
        
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
        self.max_sequence_length = max_sequence_length
        
            
        self.hidden_size = hidden_size

        # output layer
        self.output_layer = nn.Sequential(nn.Linear(self.direction*hidden_size, output_size),act_fn)

        self.crf_layer = ConditionalRandomField(num_tags= output_size, constraints=allowed_transitions(constraint_type="BIO", labels={0:'B', 1:'I', 2:'O'}))

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

        # out = self.loss_fn_crf(out,y,batch, self.max_sequence_length)
        
        return out

    def loss_fn(self, outputs, labels, batch=8, max_sequence_length=128):
    

        num_class = outputs.shape[-1]

        batch = 8

        outputs = outputs.reshape(-1,max_sequence_length,num_class)

        flat_labels = labels.reshape(-1,num_class)
        pad_index=[1 if flat_labels[i].sum()!=0 else 0 for i in range(flat_labels.shape[0])]
        
        mask = torch.FloatTensor(pad_index)
        mask = mask.to('cuda')
        mask = mask.reshape(-1,max_sequence_length)

        labels = torch.argmax(labels, dim=-1)
        
        loss = -self.crf_layer(outputs, labels, mask) / float(batch)


        return loss

    
    
    def init_hidden(self, batch_size):
                
        torch.manual_seed(0)
        h0 = torch.randn(self.direction*self.layers, batch_size, self.hidden_size, device=self.device) 
        c0 = torch.randn(self.direction*self.layers, batch_size, self.hidden_size, device=self.device)

        return h0,c0

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
        self.max_sequence_length = max_sequence_length
        
            
        self.hidden_size = hidden_size

        # output layer
        self.output_layer = nn.Sequential(nn.Linear(self.direction*hidden_size, output_size),act_fn)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

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

        # out = self.loss_fn_crf(out,y,batch, self.max_sequence_length)
        
        return out

    def loss_fn(self, outputs, labels, batch = 8):
        
        #define cross entropy loss 
        
        
        #reshape labels to give a flat vector of length batch_size*seq_len
        num_class = labels.shape[-1]
        
        # reshape label to make it similar to model output
        labels = labels.reshape(-1,num_class) 

        #get loss
        loss = self.criterion(outputs, labels.float())
        
        #get non-pad index
        non_pad_index=[i for i in range(labels.shape[0]) if labels[i].sum()!=0]
        
        #get final loss
        loss = loss[non_pad_index].mean()
        
        return loss
        

    
    
    def init_hidden(self, batch_size):
                
        torch.manual_seed(0)
        h0 = torch.randn(self.direction*self.layers, batch_size, self.hidden_size, device=self.device) 
        c0 = torch.randn(self.direction*self.layers, batch_size, self.hidden_size, device=self.device)

        return h0,c0