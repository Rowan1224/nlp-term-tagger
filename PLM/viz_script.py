import pandas as pd
from colorama import Style, Back



def print_predictions(tokens, pred_tags, true_tags):
    
    tokens = tokens.split()
    pred_tags = pred_tags.split()
    true_tags = true_tags.split()
    
    if len(tokens) != len(pred_tags):
        print(tokens)
        return " "
    
    output = []
    
    
    for t,tl,pl in zip(tokens,true_tags,pred_tags):

        if tl == pl:
            o = f"{t} {Back.GREEN}[{tl}][{pl}]{Style.RESET_ALL}"

        else:
            o = f"{t} {Back.GREEN}[{tl}]{Style.RESET_ALL}{Back.RED}[{pl}]{Style.RESET_ALL}"

        output.append(o)
        
    return " ".join(output)


def output_reader(path):
    df = pd.read_csv(path)
    df['visualizations'] = df.apply(lambda x: print_predictions(x['sent'], x['predictions'], x["sent"]), axis=1)
    return df["visualizations"].tolist()

def main():


    # Load data as pandas dataframe
    bert_test_viz = output_reader("./output/bert-base-uncased-Base/outputs.csv")
    distilbert_test_viz = output_reader("./output/distilbert-base-uncased-Base/outputs.csv")
    print(bert_test_viz[10])
    print(distilbert_test_viz[10])


if __name__ == '__main__':
    main()