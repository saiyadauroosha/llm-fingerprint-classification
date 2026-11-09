# rnn_rq2.py
#
# RQ2: Does the user's prompt help identify the LLM?

# Three experiments:
# 1. Input only     -> prompt
# 2. Output only    -> response
# 3. Input + Output -> prompt + response


import argparse
import copy
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


from torch.nn.utils.rnn import pack_padded_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
PROMPT_LEN = 128
MAX_LEN = 512
MIN_FREQUENCY = 2

##### 1. Load the fixed train, validation, and test splits #####

parser = argparse.ArgumentParser(description="Train the three RQ2 LSTM input modes.")
parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "rnn" / "rq2")
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()
if args.epochs < 1:
    parser.error("--epochs must be at least 1")

train = pd.read_parquet(args.data_dir / "train.parquet")
val = pd.read_parquet(args.data_dir / "validation.parquet")
test = pd.read_parquet(args.data_dir / "test.parquet")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
run_dir = args.output_dir / timestamp
run_dir.mkdir(parents=True, exist_ok=False)

# convert the LLM names into numbers that the model can use
label2id = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}


##### 2. Tokenizer ######
# breaks each response into smaller tokens, like words, newlines, and punctuation

def tokenize(text):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return re.findall(r"\w+|\n|[^\w\s]", text)


##### 3. Function to build vocabulary from training data ####

def build_vocab(train):
    vocab = {"<pad>": 0, "<unk>": 1}

    response_counts = Counter(
        token
        for text in train["response"]
        for token in tokenize(text)[:MAX_LEN]
    )
    for token, count in response_counts.items():
        if count >= MIN_FREQUENCY:
            vocab[token] = len(vocab)

    vocab["<sep>"] = len(vocab)

    prompt_counts = Counter(
        token
        for text in train["prompt"]
        for token in tokenize(text)[:PROMPT_LEN]
    )
    for token, count in prompt_counts.items():
        if count >= MIN_FREQUENCY:
            vocab[f"prompt:{token}"] = len(vocab)

    return vocab


##### 4. Convert text to numbers #####
def encode(tokens, vocab, max_len):
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_len]]
    if not ids:
        ids = [vocab["<unk>"]]
    return ids + [vocab["<pad>"]] * (max_len - len(ids))


##### 5. Dataset #####
# prepares the texts and labels so pyTorch can load them in batches during training and testing

class TextDataset(Dataset):

    def __init__(self, df, column, vocab, max_len):

        # convert the text into numbers using the vocab

        self.x = [
            encode(tokens, vocab, max_len)
            for tokens in df[column]
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


#### RQ2 Inputs #####

vocab = build_vocab(train)

for frame in (train, val, test):
    frame["input_prompt"] = frame["prompt"].map(
        lambda text: [f"prompt:{token}" for token in tokenize(text)[:PROMPT_LEN]]
    )
    frame["input_response"] = frame["response"].map(
        lambda text: tokenize(text)[:MAX_LEN]
    )
    frame["input_both"] = [
        prompt + ["<sep>"] + response
        for prompt, response in zip(frame["input_prompt"], frame["input_response"])
    ]

#### Train and evaluate the model

def run_experiment(train, val, test, column, vocab, run_dir, epochs):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    max_len = {
        "input_prompt": PROMPT_LEN,
        "input_response": MAX_LEN,
        "input_both": PROMPT_LEN + 1 + MAX_LEN,
    }[column]
    train_ds = TextDataset(train, column, vocab, max_len)
    val_ds = TextDataset(val, column, vocab, max_len)
    test_ds = TextDataset(test, column, vocab, max_len)
    train_dl = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        generator=torch.Generator().manual_seed(SEED),
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        generator=torch.Generator().manual_seed(SEED),
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

    best_val_accuracy = -1.0
    best_model = None

    for epoch in range(epochs):

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

    # Macro Precision 
    macro_precision = precision_score(
        test_labels, 
        test_predictions, 
        average="macro", 
        zero_division=0
    )

    #Macro Recall
    macro_recall = recall_score(
        test_labels, 
        test_predictions,
        average="macro", 
        zero_division=0
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
    print("Macro Precision:", macro_precision) 
    print("Macro Recall:", macro_recall)
    print("Macro F1:", macro_f1)
    print("Confusion Matrix:")
    print(cm)

    id2label = {index: family for family, index in label2id.items()}
    predictions = test[["sample_id", "family"]].copy()
    predictions["predicted_family"] = [id2label[index] for index in test_predictions]
    predictions.to_csv(run_dir / f"{column}_test_predictions.csv", index=False)

    return accuracy, macro_f1


#### Run and save RQ2 experiments

results_file = run_dir / "rq2_lstm_results.json"

all_results = {}

for column in ["input_prompt", "input_response", "input_both"]:

    accuracy, macro_f1 = run_experiment(train, val, test, column, vocab, run_dir, args.epochs)

    all_results[column] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

with open(results_file, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\nRQ2 results saved to: {results_file}")
