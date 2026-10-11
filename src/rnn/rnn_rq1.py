# RQ1: Identify the LLM from its response using an LSTM

import re
from collections import Counter
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report


import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import copy 
import random
import numpy as np 
from torch.nn.utils.rnn import pack_padded_sequence 
from sklearn.metrics import precision_score, recall_score

random. seed (42)
np.random.seed(42)
torch.manual_seed (42)




# 1. Load the fixed train/validation/test splits
train = pd.read_parquet("data/splits/train.parquet")
val = pd.read_parquet("data/splits/validation.parquet")
test = pd.read_parquet("data/splits/test.parquet")


#Create a mapping from LLM family names to integer labels
label2id = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}


# 2. Tokenize

def tokenize(text):
    # Keep words, newlines, and punctuation/markdown symbols
    return re.findall(r"\w+|\n|[^\w\s]", str(text))


# 3. Build vocabulary from TRAINING data only
#this counts how many times each token appears in the training data 
counts = Counter(
    token
    for response in train["response"]
    for token in tokenize(response)
)

vocab = {
    "<pad>": 0,
    "<unk>": 1
}

#Add tokens that appear at least twice to the vocabulary 
for token, count in counts.items():
    if count >= 2:
        vocab[token] = len(vocab)

#print("Vocabulary size:", len(vocab))


# 4. Convert text into numbers

MAX_LEN = 512



def encode(text):
    tokens = tokenize(text)

    ids = [
        vocab.get(token, 1)
        for token in tokens
    ]

    # Cut responses that are too long
    ids = ids[:MAX_LEN]

    # Add padding if response is shorter than MAX_LEN
    ids += [0] * (MAX_LEN - len(ids))

    return ids


# 5. Create Dataset

class TextDataset(Dataset):

    #Convert the text responses and labels into numerical format for training
    def __init__(self, df):
        self.x = [encode(response) for response in df["response"]]
        self.y = [label2id[family] for family in df["family"]]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            torch.tensor(self.x[index], dtype=torch.long),
            torch.tensor(self.y[index], dtype=torch.long)
        )


#This creates datasets for the training, validation, and test splits
train_ds = TextDataset(train)
val_ds = TextDataset(val)
test_ds = TextDataset(test)


#DataLoader groups examples into batches and shuffles the training data
# A batch size of 64 means the model processes 64 responses at a time
train_dl = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True
)

#The validation and the test data are not shuffled
val_dl = DataLoader(
    val_ds,
    batch_size=64
)

test_dl = DataLoader(
    test_ds,
    batch_size=64
)


# 6. LSTM Model

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

        # Convert word IDs into vectors
        #This allows the model to learn useful representations of the words/tokens
        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=0
        )

        # LSTM reads the response one token at a time
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        #Dropout randomly removes some of values during training to help reduce overfitting
        self.dropout = nn.Dropout(dropout)

        # Final prediction: GPT, Claude, or Gemini
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


model = TextLSTM(len(vocab))

print(model)


# 7. Train

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

#This measures how different the model's predictions are from the actual labels
loss_fn = nn.CrossEntropyLoss()

#Train the model for 10 complete passes through the training data
EPOCHS = 10

#keep track of the best seen so far
best_val_accuracy = 0
best_model_state = None



for epoch in range(EPOCHS):

    # Training

    #The model is in training mode so this activates dropout
    model.train()

    #Keep track of the total training loss for each epoch
    total_loss = 0

    #Go through the training data one batch at a time
    for x, y in train_dl:
        
        #Clear gradients from previous batch so no overlap
        optimizer.zero_grad()

        #The model makes its predictions for the current batch
        predictions = model(x)

        #Calculate how different the predictions are from actual
        loss = loss_fn(predictions, y)

        #Calculate gradients from each in the model
        loss.backward()

        #Update the model's parameters based on the gradients
        optimizer.step()

        #Keep track of total loss
        total_loss += loss.item()


    # Validation

    model.eval()



    val_predictions = []
    val_labels = []

    with torch.no_grad():

        #Go through validation data one by one
        for x, y in val_dl:

            #Get the models predictions, select class with highest score and store the predictions and also correct labels
            predictions = model(x)

            predicted_labels = predictions.argmax(dim=1)

            val_predictions.extend(
                predicted_labels.tolist()
            )

            val_labels.extend(
                y.tolist()
            )


    #Calculate the accuracy 
    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {total_loss / len(train_dl):.4f} "
        f"- Val Accuracy: {val_accuracy:.4f}"
    )


    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = copy.deepcopy(model.state_dict())


# 8. Load the best model

model.load_state_dict(best_model_state)



# 9. Final test evaluation

#Evaluates the trained model on unseen data
#It also stores the predictions and the correct labels for analysis of accuracy 
model.eval()

test_predictions = []
test_labels = []

with torch.no_grad():

    for x, y in test_dl:

        predictions = model(x)

        predicted_labels = predictions.argmax(dim=1)

        test_predictions.extend(
            predicted_labels.tolist()
        )

        test_labels.extend(
            y.tolist()
        )


# Accuracy

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

# Macro Precision 
test_macro_precision = precision_score(
    test_labels, 
    test_predictions, 
    average="macro", 
    zero_division=0
)

#Macro Recall
test_macro_recall = recall_score(
    test_labels, 
    test_predictions,
    average="macro", 
    zero_division=0
)



# Macro F1

test_macro_f1 = f1_score(
    test_labels,
    test_predictions,
    average="macro"
)


# Confusion matrix

cm = confusion_matrix(
    test_labels,
    test_predictions
)


print("\n===== RQ1 TEST RESULTS =====")

print("Test Accuracy:", test_accuracy)


print("Test Macro Precision:", test_macro_precision) 

print("Test Macro Recall:", test_macro_recall)

print("Test Macro F1:", test_macro_f1)

print("\nConfusion Matrix:")
print(cm)


# 10. Save RNN results

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "rnn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

results = {
    "model": "LSTM",
    "test_accuracy": test_accuracy,
    "test_macro_precision": test_macro_precision,
    "test_macro_recall": test_macro_recall,
    "test_macro_f1": test_macro_f1,
    "confusion_matrix": cm.tolist()
}

results_file = RESULTS_DIR / "rq1_lstm_results.json"

with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {results_file}")
