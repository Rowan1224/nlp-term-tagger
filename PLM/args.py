import argparse

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", default=4, type=int, help="Provide the number of batch"
    )
    parser.add_argument(
        "-epoch", "--epoch", default=10, type=int, help="Provide the number of epochs"
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-5,
        type=float,
        help="Provide the learning rate",
    )
    parser.add_argument(
        "-l",
        "--seq_length",
        default=128,
        type=int,
        help="define Max sequence length",
    )


    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="pre-trained model",
    )

    parser.add_argument(
        "-c",
        "--use_crf",
        action='store_true',
        help="Pass if you train pre-trained model with CRF layer",
    )

    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        type=str,
        default="500",
        help="Define Best checkpoint",
    )

    args = parser.parse_args()
    return args

 
    