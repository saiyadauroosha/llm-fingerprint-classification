# rnn_rq2.py
#
# RQ2: Does the user's prompt help identify the LLM?

# Three experiments:
# 1. Input only     -> prompt
# 2. Output only    -> response
# 3. Input + Output -> prompt + response


import re
import copy
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


##### 1. Load the fixed train, validation, and test splits #####


train = pd.read_parquet("data/splits/train.parquet")
val = pd.read_parquet("data/splits/validation.parquet")
test = pd.read_parquet("data/splits/test.parquet")

# convert the LLM names into numbers that the model can use
label2id = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}


##### 2. Tokenizer ######
# breaks each response into smaller tokens, like words, newlines, and punctuation

def tokenize(text):
    return re.findall(r"\w+|\n|[^\w\s]", str(text))


##### 3. Function to build vocabulary from training data ####

## vocab is built from the training data only and do not use validation or test data to create the vocab


def build_vocab(texts):

    # counts how often each token appears in the training data
    counts = Counter(
        token
        for text in texts
        for token in tokenize(text)
    )

    # <pad> is used to fill in the empty space when a response is shorter than the maximum length
    # <unk> is used for tokens that are not in the vocab
    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }

    # only include tokens that appear at least twice
    # This keeps the vocab from getting unnecessarily large
    for token, count in counts.items():
        if count >= 2:
            vocab[token] = len(vocab)

    return vocab


##### 4. Convert text to numbers #####
# The LSTM needs numbers instead of text, so each token is replaced by its num from the vocab
# we also made the inputs the same length; longer inputs are cut off at 512 tokens
# shorter texts are filled with <pad> tokens

MAX_LEN = 512

def encode(text, vocab):

    tokens = tokenize(text)

    # convert each token into its vocab number
    # if a token is not in the vocab use <unk> = 1

    ids = [
        vocab.get(token, 1)
        for token in tokens
    ]

    # keep at most 512 tokens
    ids = ids[:MAX_LEN]

    # add padding until the input is exactly 512 tokens long
    ids += [0] * (MAX_LEN - len(ids))

    return ids


##### 5. Dataset #####
# prepares the texts and labels so pyTorch can load them in batches during training and testing

class TextDataset(Dataset):

    def __init__(self, df, column, vocab):

        # convert the text into numbers using the vocab

        self.x = [
            encode(text, vocab)
            for text in df[column]
        ]

        # convert the LLM names into numbers that the model can use
        self.y = [
            label2id[family]
            for family in df["family"]
        ]

    def __len__(self):
        return len(self.y) # return the number of samples in the dataset

    def __getitem__(self, index):

        # return the input and label for a given index as tensors
        return (
            torch.tensor(self.x[index], dtype=torch.long),
            torch.tensor(self.y[index], dtype=torch.long)
        )


##### 6. LSTM model ####

# Reads the sequence of tokens and learns to predict which LLM generated the response

class TextLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        emb_dim=200,
        hidden_dim=128,
        n_classes=3,
        dropout=0.5
    ):

        super().__init__()

        # convert each token into a vector of numbers
        # embedding that the LSTM can work with
        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=0
        )

        # the LSTM processes the sequence of embeddings
        # and keeps information about the text as it reads it
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # dropout helps reduce overfitting during training
        self.dropout = nn.Dropout(dropout)

        # final classification layer
        self.fc = nn.Linear(
            hidden_dim,
            n_classes
        )

    def forward(self, x):

        # convert token IDs into embeddings
        embedded = self.embedding(x)

        # pass the embeddings through the LSTM
        output, (hidden, cell) = self.lstm(embedded)

        # use the final hidden state for classification
        final_hidden = hidden[-1]

        # use that summary to predict the LLM family
        return self.fc(
            self.dropout(final_hidden)
        )


#### RQ2 Inputs #####

# Prompt only
train["input_prompt"] = train["prompt"]
val["input_prompt"] = val["prompt"]
test["input_prompt"] = test["prompt"]


# Response only
train["input_response"] = train["response"]
val["input_response"] = val["response"]
test["input_response"] = test["response"]


# Prompt + response
train["input_both"] = (
    train["prompt"] + " " + train["response"]
)

val["input_both"] = (
    val["prompt"] + " " + val["response"]
)

test["input_both"] = (
    test["prompt"] + " " + test["response"]
)


#### Train and evaluate the model

def run_experiment(train, val, test, column):

    # Built vocabulary from training data only
    vocab = build_vocab(train[column])

    # Create datasets

    # convert the selected text into token IDs and convert the LLM names into numeric labels
    train_ds = TextDataset(train, column, vocab)
    val_ds = TextDataset(val, column, vocab)
    test_ds = TextDataset(test, column, vocab)

    # Create loaders
    # Loaders are used to load the data in batches during training and evaluation
    train_dl = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=64
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=64
    )

    ## Create model ##

    # model takes the vocab size so it knows how many different tokens it receives as input
    model = TextLSTM(len(vocab))


    # Optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # loss function for multi-class classification
    loss_fn = nn.CrossEntropyLoss()

    ## Train ##

    # keep track of the best model based on validation accuracy

    best_val_accuracy = 0
    best_model = None

    for epoch in range(10):

        model.train() # set the model to training mode

        # go through the training data in batches
        for x, y in train_dl:

            optimizer.zero_grad() # clear the gradients from the previous step

            predictions = model(x) # Use the LSTM to make predictions

            loss = loss_fn(predictions, y) # compare the predictions to the correct labels

            loss.backward() # calculates how the model should change

            optimizer.step() # update the models weights

        # Validation

        model.eval()

        val_predictions = []
        val_labels = []

        # gradient is not needed during validation
        with torch.no_grad():

            for x, y in val_dl:

                # make predictions on the validation data
                predictions = model(x)

                predicted = predictions.argmax(dim=1) # choose the class with the highest score

                # save the predictions and the correct labels
                val_predictions.extend(
                    predicted.tolist()
                )

                val_labels.extend(
                    y.tolist()
                )

        # calculate the accuracy on the validation set
        val_accuracy = accuracy_score(
            val_labels,
            val_predictions
        )

        # save the best model if it has the highest validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model.state_dict())

    # Test using best model
    model.load_state_dict(best_model)

    model.eval()

    test_predictions = []
    test_labels = []

    with torch.no_grad():

        for x, y in test_dl:

            predictions = model(x) # make predictions on the test data

            predicted = predictions.argmax(dim=1) # choose the class with the highest score

            test_predictions.extend( # save the predictions
                predicted.tolist()
            )

            test_labels.extend(
                y.tolist()
            )

    #### calculate final test metrics #####

    accuracy = accuracy_score(
        test_labels,
        test_predictions
    )

    macro_f1 = f1_score(
        test_labels,
        test_predictions,
        average="macro"
    )

    cm = confusion_matrix(
        test_labels,
        test_predictions
    )

    #### Print results ####

    print("\n", column)
    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print("Confusion Matrix:")
    print(cm)

    return accuracy, macro_f1


run_experiment(train, val, test, "input_prompt")

run_experiment(train, val, test, "input_response")

run_experiment(train, val, test, "input_both")