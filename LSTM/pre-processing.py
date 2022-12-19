

import os
import fnmatch
import string


def get_files(path, pattern='*.final'):
    """
    It takes a path and a pattern as input, and returns a list of files that match the pattern
    
    :param path: the path to the directory containing the files
    :param pattern: the pattern of the files to be read. Default is '*.final', defaults to *.final
    (optional)
    :return: A list of files that match the pattern
    """
    
    files = []
    for root, dirnames, filenames in os.walk(path):

        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename) )


    # sort the list, to avoid mismatch in the output files
    files = sorted(files)
    
    return files



def read_data(file):
    """
    It reads in a file, and returns a list of lists, where each list is a sentence
    
    :param file: the file to be read
    :return: A list of lists of lists.
    """
    
    try:
        with open(file,'r') as file:
            lines = [(line.strip().split()) for line in file if len(line.strip().split())==2]
            
    except:
        print(f'Courrpt File: {file}')
        return []
        
    i = 0
    sentences = []
    
    for j,token in enumerate(lines):
    
    
        if len(token)>0 and token[0] =='.':
            sentences.append(lines[i:j+1])
            i = j+1
        if len(token)==1:
            print(file)
        else:
            continue
            
    return sentences
                

def prepare_data(files):
    """
    - Reads the data from the files
    - Creates a list of sentences and a list of labels
    - Returns the lists
    
    :param files: list of files to read
    """
    
    mx_length = 0
    data = []
    for file in files:
        data.extend(read_data(file))
    
    sens = []
    labels = []
    for item in data:
        
        tokens = []
        label = []
        for entry in item:
            if len(entry[0]) == 1 and entry[0] in string.punctuation:
                continue
            else:
                if '0' in entry[1]:
                    entry[1] = entry[1].replace('0','O')
                tokens.append(entry[0])
                label.append(entry[1])
        
        assert len(label)==len(tokens)
        
        if len(tokens) > 100:
            continue
        
        sens.append(" ".join(tokens))
        labels.append(" ".join(label))
        
        mx_length = max(mx_length, len(tokens))
        

        
    print(f"Max Length: {mx_length}")
    
    if mx_length> 100:
        print(file)
        
    return sens, labels
        



def write_files(sens,labels,path):
    
    filenameSen = os.path.join(path,'sentences.txt')
    filenameLabel = os.path.join(path,'labels.txt')
    
    with open(filenameSen,'w') as file:
        lines = "\n".join(sens)
        file.write(lines)
        
    
    with open(filenameLabel,'w') as file:
        lines = "\n".join(labels)
        file.write(lines)
    
    


def create_vocab(sentences):    

    word_string = ' '.join(sentences)
        
    vocab = set(word_string.split())
    return list(vocab)



if __name__ == '__main__':
    
    train_path = '../Dataset/train'
    dev_path = '../Dataset/dev'
    test_path = '../Dataset/test'


    train_files = get_files(train_path)
    dev_files = get_files(dev_path)
    test_files = get_files(test_path)


    train_sens, train_labels = prepare_data(train_files)
    dev_sens, dev_labels = prepare_data(dev_files)
    test_sens, test_labels = prepare_data(test_files)



    write_files(train_sens, train_labels, train_path)
    write_files(dev_sens, dev_labels, dev_path)
    write_files(test_sens, test_labels, test_path)



    vocab = create_vocab(train_sens)

    with open('./vocab.txt','w') as file:
            lines = "\n".join(vocab)
            file.write(lines)






