from torch.utils.data import Dataset
import torch


class AnnotatedDataset(Dataset):

    def __init__(self, labels, sents, unique_labels, tokenizer):
        self.tokenizer = tokenizer
        self.labels = [[lab.upper() for lab in label] for label in labels]
        self.tokenized = [self.tokenizer(sent, padding="max_length", max_length=512, truncation=True, return_tensors="pt") for sent in sents]
        self.labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        self.ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}                
        self.aligned_labels = self.align_labels()

    def __len__(self):
        return len(self.aligned_labels)

    def __getitem__(self, idx):
        return self.sent_data(idx), self.label_data(idx)


    def sent_data(self, idx):
        return self.tokenized[idx]

    def label_data(self, idx):
        return torch.LongTensor(self.aligned_labels[idx])

    def align_labels(self):
        word_ids = [token_text['input_ids'][0] for token_text in self.tokenized]        
        
        pre_alignment = list(zip(word_ids, self.labels))           
    
        aligned_labels = []
        for word_idx, labels in pre_alignment:
            previous_idx = None
            label_ids = []
            for idx in word_idx:        

                if idx == 0:
                    label_ids.append(-100)
                
                elif idx != previous_idx:
                    try:
                        label_ids.append(self.labels_to_ids[labels[idx]])
                    except IndexError:
                        label_ids.append(-100)
                        continue
                else:
                    #label_ids.append(self.labels_to_ids[labels[idx]])
                    label_ids.append(-100)
                previous_idx = idx
            
        
            #print(self.tokenizer.convert_ids_to_tokens(tito))
            
            #print(label_ids[:30])
            print(self.tokenizer.convert_ids_to_tokens(word_idx))  
            raise SystemExit
            assert len(label_ids) == 512, "alignment failing, len should be 512"

            aligned_labels.append(label_ids)
        return aligned_labels
