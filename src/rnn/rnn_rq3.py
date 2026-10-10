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

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


import random
import numpy as np 
from torch.nn.utils.rnn import pack_padded_sequence

# Set random seeds for reproducibility
random.seed (42)
np.random.seed(42)
torch.manual_seed(42)

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
        #Find each response's real length so padding doesn't affect final state
        lengths = (x != 0). sum(dim=1).clamp(min=1).cpu()
        # x: (batch size, sequence length, embedding dimension)
        embedded = self.embedding(x)

        # LSTM processes the sequence
        #This produces 
        #-output: hidden states for every token
        #-hidden: final hidden state
        #-cell: final cell state

        # NEW: Pack padded sequence so the LSTM stops at the actual text end
        packed = pack_padded_sequence (
            embedded, 
            lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        output, (hidden, cell) = self.lstm(packed)

        # Use the final hidden state
        final_hidden = hidden[-1]

        # Predict the LLM family
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
    
    seen_test_data = test[test["domain"] != held_out_domain].copy()


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

    seen_test_ds = TextDataset(seen_test_data, vocab)

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
    seen_test_dl = DataLoader(seen_test_ds, batch_size=64)

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

# NEW: Helper functions to evaluate loaders and print full metrics for both splits
    model.load_state_dict(best_model)
    model.eval()

    def evaluate(loader):
        predictions_all = []
        labels_all = []
        with torch.no_grad():
            for x, y in loader:
                predictions = model(x)
                predicted = predictions.argmax(dim=1)
                predictions_all.extend(predicted.tolist())
                labels_all.extend(y.tolist())
        return labels_all, predictions_all

    test_labels, test_predictions = evaluate(test_dl)
    seen_labels, seen_predictions = evaluate(seen_test_dl)

    def print_metrics(title, labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        macro_precision = precision_score(labels, predictions, average="macro", zero_division=0)
        macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
        print("\n" + title)
        print("Accuracy:", accuracy)
        print("Macro Precision:", macro_precision)
        print("Macro Recall:", macro_recall)
        print("Macro F1:", macro_f1)
        print("Confusion Matrix:")
        print(cm)
        return accuracy, macro_f1

    print("Best validation accuracy:", best_val_accuracy)
    print_metrics("Held-out domain test results:", test_labels, test_predictions)
    seen_accuracy, seen_macro_f1 = print_metrics("Seen domain test results:", seen_labels, seen_predictions)

    return seen_accuracy, seen_macro_f1

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
