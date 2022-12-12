import argparse

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", default=8, type=int, help="Provide the number of batch"
    )
    parser.add_argument(
        "-epoch", "--epoch", default=10, type=int, help="Provide the number of epochs"
    )
    parser.add_argument(
        "-layers", "--layers", default=5, type=int, help="Provide the number of batch"
    )
    parser.add_argument(
        "-hidden", "--hidden_size", default=32, type=int, help="Provide the number of epochs"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.01,
        type=float,
        help="Provide the learning rate",
    )
    parser.add_argument(
        "-l",
        "--seq_length",
        default=96,
        type=int,
        help="define Max sequence length",
    )

    parser.add_argument(
        "-emb",
        "--embedding_size",
        default=100,
        type=int,
        choices=[50, 100, 200, 300],
        help="Select the model type for training (fine-tuning or domain adaption on SQuAD model)",
    )

    parser.add_argument(
        "-v",
        "--vector_path",
        type=str,
        default="./glove.6B/glove.6B.100d.txt",
        help="Word Embedding Path",
    )

    parser.add_argument(
        "-c",
        "--use_crf",
        action='store_true',
        help="Pass if you train LSTM with CRF layer",
    )

    args = parser.parse_args()
    return args

 
    