# rnn_rq3.py
#
# RQ3: Can the RNN identify the LLM on a domain that was not used during training?
#
# Experiments ran:
# 1. Hold out General domain
# 2. Hold out Code domain
# 3. Hold out Math domain
# 4. Hold out Creative Writing domain


import re
import copy
from collections import Counter

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


#### 1. load the data ####

train = pd.read_parquet("data/splits/train.parquet")
val = pd.read_parquet("data/splits/validation.parquet")
test = pd.read_parquet("data/splits/test.parquet")

label2id = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}


# 2. Tokenizer
# split text into words, punctuation, and new lines

def tokenize(text):
    return re.findall(r"\w+|\n|[^\w\s]", str(text))


#### 3. Build vocabulary from training data only ####

def build_vocab(texts):

    counts = Counter(
        token
        for text in texts
        for token in tokenize(text)
    )

    # special tokens for padding and unknown words
    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }

    # keep tokens that appear at least twice
    for token, count in counts.items():

        if count >= 2:
            vocab[token] = len(vocab)

    return vocab


## 4. convert text to numbers ###

MAX_LEN = 512


def encode(text, vocab):

    tokens = tokenize(text)

    # convert each token to its corresponding number in the vocab
    ids = [
        vocab.get(token, 1)
        for token in tokens
    ]

    # cut off responses that are too long
    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))

    return ids


#### 5. dataset ####
# convert responses and labels into a format that PyTorch can use for training and testing

class TextDataset(Dataset):

    def __init__(self, df, vocab):

        self.x = [
            encode(text, vocab)
            for text in df["response"]
        ]

        self.y = [
            label2id[family]
            for family in df["family"]
        ]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        return (
            torch.tensor(
                self.x[index],
                dtype=torch.long
            ),
            torch.tensor(
                self.y[index],
                dtype=torch.long
            )
        )


#### 6. LSTM model #####

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

        # convert the token IDs into embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=0
        )

        # process the sequence of embeddings
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        # predict the LLM family
        self.fc = nn.Linear(
            hidden_dim,
            n_classes
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        # use the final hidden state for classification
        final_hidden = hidden[-1]

        return self.fc(
            self.dropout(final_hidden)
        )


#### run one held-out-domain experiment ####


def run_experiment(train, val, test, held_out_domain):

    print("Held-out domain:", held_out_domain)
    print("____________________________________")

    # remove the held-out domain from training and validation

    train_data = train[
        train["domain"] != held_out_domain
    ].copy()

    val_data = val[
        val["domain"] != held_out_domain
    ].copy()

    # test only on the unseen domain
    test_data = test[
        test["domain"] == held_out_domain
    ].copy()

    ## build vocabulary from training data only ##

    vocab = build_vocab(
        train_data["response"]
    )

    # Create datasets

    train_ds = TextDataset(
        train_data,
        vocab
    )

    val_ds = TextDataset(
        val_data,
        vocab
    )

    test_ds = TextDataset(
        test_data,
        vocab
    )

    ### create data loaders

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

    # create model, optimizer, and loss function

    model = TextLSTM(
        len(vocab)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = nn.CrossEntropyLoss()

    # train the model and evaluate on validation set

    best_val_accuracy = 0
    best_model = None

    for epoch in range(10):

        model.train()

        for x, y in train_dl:

            optimizer.zero_grad()

            predictions = model(x)

            loss = loss_fn(
                predictions,
                y
            )

            loss.backward()

            optimizer.step()

        # check performance on the validation data
        model.eval()

        val_predictions = []
        val_labels = []

        with torch.no_grad():

            for x, y in val_dl:

                predictions = model(x)

                predicted = predictions.argmax(
                    dim=1
                )

                val_predictions.extend(
                    predicted.tolist()
                )

                val_labels.extend(
                    y.tolist()
                )

        val_accuracy = accuracy_score(
            val_labels,
            val_predictions
        )

        # save best model with the best validation accuracy so far

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model = copy.deepcopy(
                model.state_dict()
            )

    # test using the best model

    model.load_state_dict(
        best_model
    )

    model.eval()

    test_predictions = []
    test_labels = []

    with torch.no_grad():

        for x, y in test_dl:

            predictions = model(x)

            predicted = predictions.argmax(
                dim=1
            )

            test_predictions.extend(
                predicted.tolist()
            )

            test_labels.extend(
                y.tolist()
            )

    # calculate the final test accuracy, macro F1, and confusion matrix

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

    # print results

    print("Best validation accuracy:", best_val_accuracy)

    print("Test Accuracy:", accuracy)

    print("Test Macro F1:", macro_f1)

    print("Confusion Matrix:")

    print(cm)

    return accuracy, macro_f1


#### run all four RQ3 experiments ####

run_experiment(
    train,
    val,
    test,
    "general"
)

run_experiment(
    train,
    val,
    test,
    "code"
)

run_experiment(
    train,
    val,
    test,
    "math"
)

run_experiment(
    train,
    val,
    test,
    "creative_writing"
)
