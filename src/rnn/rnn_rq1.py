# RQ1: Identify the LLM from its response using an LSTM

import argparse
import hashlib
import platform
import re
from collections import Counter
import json
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
from io import BytesIO
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
MAX_LEN = 512
MIN_FREQUENCY = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001

parser = argparse.ArgumentParser(description="Train the RQ1 response-only LSTM.")
parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "rnn" / "rq1",
                    help="Parent directory for the timestamped run")
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()
if args.epochs < 1:
    parser.error("--epochs must be at least 1")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Load the fixed train/validation/test splits and hash the bytes read.
frames = {}
split_hashes = {}
for split in ("train", "validation", "test"):
    data = (args.data_dir / f"{split}.parquet").read_bytes()
    frames[split] = pd.read_parquet(BytesIO(data))
    if frames[split].empty:
        raise ValueError(f"{split}: cannot run on an empty split.")
    split_hashes[split] = hashlib.sha256(data).hexdigest()
train, val, test = frames["train"], frames["validation"], frames["test"]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_dir = args.output_dir / timestamp
output_dir.mkdir(parents=True, exist_ok=False)
print("Results folder:", output_dir, flush=True)


#Create a mapping from LLM family names to integer labels
label2id = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}
family_names = list(label2id)
family_ids = list(label2id.values())


# 2. Tokenize

def tokenize(text):
    # Keep words, newlines, and punctuation/markdown symbols
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return re.findall(r"\w+|\n|[^\w\s]", text)


# 3. Build vocabulary from TRAINING data only
# Count only the tokens that the model can see in training responses.
counts = Counter(
    token
    for response in train["response"]
    for token in tokenize(response)[:MAX_LEN]
)

vocab = {
    "<pad>": 0,
    "<unk>": 1
}

#Add tokens that appear at least twice to the vocabulary 
for token, count in counts.items():
    if count >= MIN_FREQUENCY:
        vocab[token] = len(vocab)

#print("Vocabulary size:", len(vocab))


# 4. Convert text into numbers

def encode(text):
    tokens = tokenize(text)[:MAX_LEN]

    ids = [
        vocab.get(token, 1)
        for token in tokens
    ] or [1]

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
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED)
)

#The validation and the test data are not shuffled
val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    generator=torch.Generator().manual_seed(SEED)
)

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    generator=torch.Generator().manual_seed(SEED)
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
model_args = {
    "vocab_size": len(vocab),
    "emb_dim": model.embedding.embedding_dim,
    "hidden_dim": model.lstm.hidden_size,
    "n_classes": model.fc.out_features,
    "dropout": model.dropout.p,
}

print(model)


# 7. Train

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

#This measures how different the model's predictions are from the actual labels
loss_fn = nn.CrossEntropyLoss()

# Train for 10 epochs by default.
EPOCHS = args.epochs

#keep track of the best seen so far
best_val_accuracy = -1
best_model_state = None
best_epoch = None
history = []


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


    train_loss = total_loss / len(train_dl)
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_accuracy": float(val_accuracy),
        "val_macro_f1": float(f1_score(
            val_labels, val_predictions, labels=family_ids, average="macro", zero_division=0
        )),
    })
    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {train_loss:.4f} "
        f"- Val Accuracy: {val_accuracy:.4f}"
    )


    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        best_model_state = copy.deepcopy(model.state_dict())


# 8. Load the best model

model.load_state_dict(best_model_state)

training = {
    "mode": "response",
    "max_len": MAX_LEN,
    "vocab_source": "training_response_prefix",
    "best_epoch": best_epoch,
    "selection_metric": "validation_accuracy",
    "best_val_accuracy": float(best_val_accuracy),
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "min_frequency": MIN_FREQUENCY,
    "epochs": EPOCHS,
}
checkpoint = {
    "model_state": best_model_state,
    "model_args": model_args,
    "vocab": vocab,
    "label2id": label2id,
    **training,
    "training_sample_ids": train["sample_id"].astype(str).tolist(),
    "validation_sample_ids": val["sample_id"].astype(str).tolist(),
}
torch.save(checkpoint, output_dir / "best_model.pt")
pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

metadata = {
    "experiment": "RQ1 LSTM response-only",
    "script": "src/rnn/rnn_rq1.py",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "device": "cpu",
    "packages": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": version("scikit-learn"),
        "pyarrow": version("pyarrow"),
    },
    "model_args": model_args,
    "training": training,
    "split_counts": {name: len(frame) for name, frame in frames.items()},
    "split_sha256": split_hashes,
}
(output_dir / "run_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
)


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
    labels=family_ids,
    average="macro",
    zero_division=0
)

#Macro Recall
test_macro_recall = recall_score(
    test_labels,
    test_predictions,
    labels=family_ids,
    average="macro",
    zero_division=0
)



# Macro F1

test_macro_f1 = f1_score(
    test_labels,
    test_predictions,
    labels=family_ids,
    average="macro",
    zero_division=0
)


# Confusion matrix

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=family_ids
)


print("\n===== RQ1 TEST RESULTS =====")

print("Test Accuracy:", test_accuracy)


print("Test Macro Precision:", test_macro_precision) 

print("Test Macro Recall:", test_macro_recall)

print("Test Macro F1:", test_macro_f1)

print("\nConfusion Matrix:")
print(cm)


# 10. Save test metrics, confusion matrix, and per-example predictions.
report = classification_report(
    test_labels, test_predictions, labels=family_ids, target_names=family_names,
    output_dict=True, zero_division=0
)
metrics = {
    "accuracy": float(test_accuracy),
    "macro_precision": float(test_macro_precision),
    "macro_recall": float(test_macro_recall),
    "macro_f1": float(test_macro_f1),
    "n_examples": len(test),
    "classification_report": report,
}
(output_dir / "test_metrics.json").write_text(
    json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
)
pd.DataFrame(
    cm,
    index=pd.Index(family_names, name="actual"),
    columns=family_names
).to_csv(output_dir / "test_confusion_matrix.csv")

# The test loader preserves the original row order.
predictions = test[["sample_id", "family"]].copy()
predictions["predicted_family"] = [family_names[index] for index in test_predictions]
predictions.to_csv(output_dir / "test_predictions.csv", index=False)

print("\nSaved model and results to:", output_dir)
