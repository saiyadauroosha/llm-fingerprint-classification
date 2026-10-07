# rnn_rq4.py
#
# RQ4: What characteristics distinguish different LLMs?
#
# Four experiments:
# 1. Full response
# 2. Remove formatting
# 3. Remove punctuation
# 4. Length-matched responses


import re
import copy
from pathlib import Path
from collections import Counter

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score


# 1. Load the data

# Find the project folder automatically.
# This allows the script to work whether it is run
# from the project root or from src/rnn.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "splits"


train = pd.read_parquet(
    DATA_DIR / "train.parquet"
)

val = pd.read_parquet(
    DATA_DIR / "validation.parquet"
)

test = pd.read_parquet(
    DATA_DIR / "test.parquet"
)


label2id = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}


# 2. Tokenizer

def tokenize(text):

    return re.findall(
        r"\w+|\n|[^\w\s]",
        str(text)
    )


# 3. Build vocabulary from training data only

def build_vocab(texts):

    counts = Counter(
        token
        for text in texts
        for token in tokenize(text)
    )

    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }

    for token, count in counts.items():

        if count >= 2:

            vocab[token] = len(vocab)

    return vocab


# 4. Convert text to numbers

MAX_LEN = 512


def encode(text, vocab):

    tokens = tokenize(text)

    ids = [
        vocab.get(token, 1)
        for token in tokens
    ]

    ids = ids[:MAX_LEN]

    ids += [0] * (
        MAX_LEN - len(ids)
    )

    return ids


# 5. Dataset

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


# 6. LSTM model

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

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            hidden_dim,
            n_classes
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(
            embedded
        )

        final_hidden = hidden[-1]

        return self.fc(
            self.dropout(final_hidden)
        )


# 7. Response transformations


# Experiment 1:
# Full response: this uses the full response text as-is, without any modifications.

def full_response(text):

    return str(text)


# Experiment 2:
# Remove formatting
#
# Removes:
# - newlines
# - common Markdown formatting symbols
# - extra whitespace

def remove_formatting(text):

    text = str(text)

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"[*_`#>-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# Experiment 3:
# Remove punctuation from the responses, this tests with 
# punctuation had an effect in how the RNN classifies the responses

def remove_punctuation(text):

    text = str(text)

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# Experiment 4:
# Length matching
#
# We find a common word limit based on the shortest
# maximum response length across the three splits.
#
# Every response is then truncated to that same
# maximum number of words.

def length_match(
    train,
    val,
    test
):

    train = train.copy()

    val = val.copy()

    test = test.copy()

    train_lengths = (
        train["response"]
        .astype(str)
        .apply(
            lambda x: len(x.split())
        )
    )

    val_lengths = (
        val["response"]
        .astype(str)
        .apply(
            lambda x: len(x.split())
        )
    )

    test_lengths = (
        test["response"]
        .astype(str)
        .apply(
            lambda x: len(x.split())
        )
    )

    common_length = min(
        train_lengths.max(),
        val_lengths.max(),
        test_lengths.max(),
        MAX_LEN
    )

    print(
        "Common maximum word length:",
        common_length
    )

    def match_length(text):

        words = str(text).split()

        return " ".join(
            words[:common_length]
        )

    train["response"] = (
        train["response"]
        .apply(match_length)
    )

    val["response"] = (
        val["response"]
        .apply(match_length)
    )

    test["response"] = (
        test["response"]
        .apply(match_length)
    )

    return (
        train,
        val,
        test
    )


# 8. Run one RQ4 experiment

def run_experiment(
    train,
    val,
    test,
    experiment_name,
    transform
):

    print("\n===================================")
    print(experiment_name)
    print("===================================")

    # Make copies so the original data is not changed

    train_data = train.copy()

    val_data = val.copy()

    test_data = test.copy()

    # Apply transformation

    train_data["response"] = (
        train_data["response"]
        .apply(transform)
    )

    val_data["response"] = (
        val_data["response"]
        .apply(transform)
    )

    test_data["response"] = (
        test_data["response"]
        .apply(transform)
    )

    print(
        "Training examples:",
        len(train_data)
    )

    print(
        "Validation examples:",
        len(val_data)
    )

    print(
        "Test examples:",
        len(test_data)
    )

    # Build vocabulary from training data ONLY

    vocab = build_vocab(
        train_data["response"]
    )

    print(
        "Vocabulary size:",
        len(vocab)
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

    # Create data loaders

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

    # Create model

    model = TextLSTM(
        len(vocab)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = nn.CrossEntropyLoss()

    # Train

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

        # Validation

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

        print(
            "Epoch",
            epoch + 1,
            "- Val Accuracy:",
            round(val_accuracy, 4)
        )

        # Save best model

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_model = copy.deepcopy(
                model.state_dict()
            )

    # Test using the best model

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

    # Calculate metrics

    accuracy = accuracy_score(
        test_labels,
        test_predictions
    )

    macro_f1 = f1_score(
        test_labels,
        test_predictions,
        average="macro"
    )

    # Print results

    print(
        "\nBest validation accuracy:",
        round(best_val_accuracy, 4)
    )

    print(
        "Test Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Test Macro F1:",
        round(macro_f1, 4)
    )

    return (
        accuracy,
        macro_f1
    )


# 9. RQ4 Experiment 1
# Full response

run_experiment(
    train,
    val,
    test,
    "Experiment 1: Full response",
    full_response
)


# 10. RQ4 Experiment 2
# Remove formatting

run_experiment(
    train,
    val,
    test,
    "Experiment 2: Remove formatting",
    remove_formatting
)


# 11. RQ4 Experiment 3
# Remove punctuation

run_experiment(
    train,
    val,
    test,
    "Experiment 3: Remove punctuation",
    remove_punctuation
)


# 12. RQ4 Experiment 4
# Length-matched responses

train_matched, val_matched, test_matched = (
    length_match(
        train,
        val,
        test
    )
)


run_experiment(
    train_matched,
    val_matched,
    test_matched,
    "Experiment 4: Length matched",
    full_response
)