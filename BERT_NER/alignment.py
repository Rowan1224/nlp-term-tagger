from torch.utils.data import Dataset
import torch


class AnnotatedDataset(Dataset):

    def __init__(self, labels, sents, labels_to_ids, tokenizer):
        self.tokenizer = tokenizer
        self.labels = [[lab.upper() for lab in label] for label in labels]
        self.tokenized = [self.tokenizer(sent, is_split_into_words=True, padding="max_length", max_length=128, truncation=True,
                          return_offsets_mapping=True) for sent in sents]
        self.labels_to_ids = labels_to_ids
        self.aligned_labels = self.align_labels()

    def __len__(self):
        return len(self.aligned_labels)

    def __getitem__(self, idx):
        return self.sent_data(idx), self.label_data(idx)


    def sent_data(self, idx):
        return {k: torch.LongTensor(v) for k,v in self.tokenized[idx].items()}

    def label_data(self, idx):
        return torch.LongTensor(self.aligned_labels[idx])

    def align_labels(self):
        word_ids = [token_text['offset_mapping'] for token_text in self.tokenized]        

        pre_alignment = zip(word_ids, self.labels)         
    
        aligned_labels = []
        for word_idx, labels in pre_alignment:
            # previous_idx = None
            
            label_ids = []
            i = 0
            for st, end in word_idx:        


                if st == 0 and end != 0:
                    label_ids.append(self.labels_to_ids[labels[i]])
                    i+=1

                else:
                    #label_ids.append(self.labels_to_ids[labels[idx]])
                    label_ids.append(-100)

            

            assert len(label_ids) == 128, "alignment failing, len should be 512"


            
            
            aligned_labels.append(label_ids)

        # for a, b in zip(aligned_labels[0],self.tokenizer.convert_ids_to_tokens(self.tokenized[0]["input_ids"])):
        #     print(a, b)
      

        # raise SystemExit


        return aligned_labels
