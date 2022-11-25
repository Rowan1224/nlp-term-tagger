from torch.utils.data import Dataset
import torch
import numpy as np

class EntityDataset(Dataset):

    def __init__(self, data_dir, vector_path, emb_dimension = 100, max_seq_len =96, device='cpu'):
        """Initialize the attributes of the object of the class."""
        
        # data directory
        self.data_dir = data_dir
        
        # load text dataset  
        self.sentences = self._read_data(data_dir, 'sentences')

        # load text dataset  
        self.labels = self._read_data(data_dir, 'labels')  
        
        # load the glove embedding
        self.vector_path = vector_path
        
        # set the embedding dimension 50/100/300
        self.emb_dimension = emb_dimension
        
        
        # set the maximum sequence length or max tweet length
        self.max_seq_len = max_seq_len
        
        # create the vocabulary from the dataset
        self.vocab = sorted(self._create_vocabulary())

        #device
        self.device = device
        
        
        # map word or tokens to index 
        self.word_to_index = {word: idx+1 for idx, word in enumerate(sorted(self.vocab))}
        
        # set pad token index to 0 and unk token index last of vocab
        self.word_to_index['[PAD]'] = 0
        self.word_to_index['[UNK]'] = len(self.vocab)+1
        
        # define the entitly labels to index values
        self.label_to_index = {'B':0, 'I':1, 'O':2, '[PAD]':-1} 

        
        # create the embedding vector
        self.word_embeddings = self._create_embedding()


        
       
        

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.sentences)

    def __getitem__(self, index):
        """Return a data sample for a given index, along with the lable of the corresponding tweet"""
        
        
        # - get the data sample corresponding to 'index' (use the list 'self.image_path_list')
        data_sample = self.sentences[index]
        label = self.labels[index]
        
        # tokenize the sentence and label
        tokens = self._tokenize_text(data_sample)
        labels = self._tokenize_text(label)

        # use the word_to_index mapping to transform the tokens into indices and save them into an IntTensor
        x = torch.IntTensor([self.word_to_index[word] 
                             if word in self.word_to_index 
                             else self.word_to_index["[UNK]"] 
                             for word in tokens])
        
        # transform the variable to cuda or cpu
        x = x.to(self.device)
        
        
        
        
        # get the index-th label and store it into a FloatTensor
        y = [self._label_map(l) for l in labels]
        y = torch.IntTensor(torch.stack(y))
        # transform the variable to cuda or cpu
        y = y.to(self.device)
        # stores the text indices and the label into a dictionary
        features = {'token_ids': x, 'labels': y}
        
        
        return features

    
    def _create_embedding(self):
        
        """create a matrix containing word vectors"""

        # load the glove embedding to a dict. token is the key and value is the vector
        embeddings_index = {}
        with open(self.vector_path,'r') as file:
            embeddings_index = {line.split()[0]: np.asarray(line.split()[1], dtype='float32') for line in file}

        # create the embedding matrix. keep the words that only present in the dataset. 
        # each row represent one vector
        # row index is the word map index
        embedding_matrix = np.zeros((len(self.word_to_index) + 2, self.emb_dimension))
        for word, i in self.word_to_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        embedding_matrix[len(self.vocab)+1] = torch.randn(self.emb_dimension)
                
        return torch.tensor(embedding_matrix, device=self.device)
        
        
    def _create_vocabulary(self):
        """Create a vocabulary of unique words from the given text files."""
        
        path = 'vocab.txt'
        with open(path, 'r') as file:
            vocab = [line.strip() for line in file]

        return list(vocab)

    def _tokenize_text(self, line):
        """
        Remove non-characters from the text and pads the text to max_seq_len.
        *!* Padding is necessary for ensuring that all text_files have the same size
        *!* This is required since DataLoader cannot handle tensors of variable length

        Return a list of all tokens in the text
        """

        tokens = line.split()
        for i in range(self.max_seq_len - len(tokens)):
            tokens.append('[PAD]')
        return tokens
    
    def _label_map(self,label,class_num=3):
        
        """ convert to labels to one hot vectors"""
        
        one_hot = torch.zeros(class_num, dtype=torch.int32)
        
        idx = self.label_to_index[label.upper()]
        if idx!=-1:
            one_hot[idx] = 1
        
        return one_hot
            
            
    
    def _read_data(self, path, pattern):
        
        """ read txt file and return as list of strings"""
        
        path = f'{path}/{pattern}.txt'
        with open(path, 'r') as file:
            data = [line.strip() for line in file]

        
        return data
         
