# Set Up Environment

Execute ``setup.sh`` file to create a virtual environment and install the required packages

# Run Exmperiment

Execute ``run.sh`` file to train and evaluate models:

Command: ``run.sh [bert/lstm/distilbert] [crf/base]``

Example:

- LSTM: ``run.sh lstm base`` # Please download GloVe word embedding vectors and place it in the LSTM folder. It should be stored in a folder called "glove.6B"

- BERT+CRF: ``run.sh bert crf``

- DistilBERT+CRF: ``run.sh distilbert crf``

- Majority Baseline: ``run.sh``

# Resources

- [Download word vectors from GloVe](https://nlp.stanford.edu/projects/glove/)
