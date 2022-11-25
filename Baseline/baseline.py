from sklearn.metrics import classification_report


def read_data(path, pattern):

    """ read txt file and return as list of strings"""

    path = f'{path}/{pattern}.txt'
    with open(path, 'r') as file:
        data = [line.strip() for line in file]


    return data
         

if __name__=='__main__':


    sentences = read_data('../Dataset/test', 'sentences')
    labels = read_data('../Dataset/test', 'labels')


    true = []
    preds = []
    true_span = []
    preds_span = []
    for label in labels:
        
        t = label.upper().split()
        true.extend(t)
        p = ['O' for i in range(len(label.split()))]
        preds.extend(p)
        preds_span.append(p)
        true_span.append(t)
            
            
    print("#"*50)
    print('Token Level Evaluation')
    print(classification_report(true,preds,zero_division=0))
    print("#"*50)
