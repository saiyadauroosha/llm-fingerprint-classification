# RQ1: Identify the LLM from its response using a CNN
 
import re

import copy

from collections import Counter
 
import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
 
 
# 1. Load the fixed train/validation/test splits

# Run this script from your project folder.
 
train = pd.read_parquet("data/splits/train.parquet")

val = pd.read_parquet("data/splits/validation.parquet")

test = pd.read_parquet("data/splits/test.parquet")
 
torch.manual_seed(42)
 
# Convert family names into integer labels.

label2id = {

    "GPT": 0,

    "Claude": 1,

    "Gemini": 2

}
 
family_names = list(label2id.keys())
 
 
# 2. Tokenize
 
def tokenize(text):

    # Keep words, newlines, punctuation, and Markdown symbols.

    return re.findall(r"\w+|\n|[^\w\s]", str(text))
 
 
# 3. Build vocabulary from TRAINING data only
 
counts = Counter(

    token

    for response in train["response"]

    for token in tokenize(response)

)
 
vocab = {

    "<pad>": 0,

    "<unk>": 1

}
 
# Include tokens that occur at least twice.

for token, count in counts.items():

    if count >= 2:

        vocab[token] = len(vocab)
 
print("Vocabulary size:", len(vocab))
 
 
# 4. Convert text into numbers
 
MAX_LEN = 512
 
def encode(text):

    tokens = tokenize(text)[:MAX_LEN]
 
    # Unknown tokens receive ID 1.

    ids = [vocab.get(token, 1) for token in tokens]
 
    # Handle an empty response.

    if not ids:

        ids = [1]
 
    # Record the actual length before adding padding.

    length = len(ids)
 
    # Pad shorter responses to MAX_LEN.

    ids += [0] * (MAX_LEN - length)
 
    return ids, length
 
 
# 5. Create Dataset and DataLoaders
 
class TextDataset(Dataset):
 
    def __init__(self, df):

        encoded = [encode(response) for response in df["response"]]
 
        self.x = [item[0] for item in encoded]

        self.lengths = [item[1] for item in encoded]

        self.y = [label2id[family] for family in df["family"]]
 
    def __len__(self):

        return len(self.y)
 
    def __getitem__(self, index):

        return (

            torch.tensor(self.x[index], dtype=torch.long),

            torch.tensor(self.lengths[index], dtype=torch.long),

            torch.tensor(self.y[index], dtype=torch.long)

        )
 
 
train_ds = TextDataset(train)

val_ds = TextDataset(val)

test_ds = TextDataset(test)
 
train_dl = DataLoader(

    train_ds,

    batch_size=64,

    shuffle=True

)
 
val_dl = DataLoader(

    val_ds,

    batch_size=64,

    shuffle=False

)
 
test_dl = DataLoader(

    test_ds,

    batch_size=64,

    shuffle=False

)
 
 
# 6. CNN Model
 
class TextCNN(nn.Module):
 
    def __init__(

        self,

        vocab_size,

        emb_dim=200,

        num_filters=100,

        kernel_sizes=(3, 4, 5),

        n_classes=3,

        dropout=0.5

    ):

        super().__init__()
 
        self.kernel_sizes = kernel_sizes
 
        # Convert token IDs into word vectors.

        self.embedding = nn.Embedding(

            vocab_size,

            emb_dim,

            padding_idx=0

        )
 
        # Different kernel sizes detect different local token patterns.

        self.convs = nn.ModuleList([

            nn.Conv1d(

                in_channels=emb_dim,

                out_channels=num_filters,

                kernel_size=k

            )

            for k in kernel_sizes

        ])
 
        self.dropout = nn.Dropout(dropout)
 
        # Three kernel sizes × 100 filters = 300 features.

        self.fc = nn.Linear(

            num_filters * len(kernel_sizes),

            n_classes

        )
 
    def forward(self, x, lengths):
 
        # Shape: [batch, sequence length, embedding dimension]

        embedded = self.embedding(x)
 
        # Conv1d expects: [batch, embedding dimension, sequence length]

        embedded = embedded.transpose(1, 2)
 
        pooled_features = []
 
        for kernel_size, conv in zip(self.kernel_sizes, self.convs):
 
            # Apply convolution and ReLU.

            features = torch.relu(conv(embedded))
 
            # Count valid windows before padding starts.

            # A very short response gets one zero-padded window.

            valid_windows = (lengths - kernel_size + 1).clamp(min=1)
 
            positions = torch.arange(

                features.size(2),

                device=features.device

            )
 
            padding_mask = (

                positions.unsqueeze(0) >= valid_windows.unsqueeze(1)

            )
 
            # Prevent padding from being selected by max pooling.

            features = features.masked_fill(

                padding_mask.unsqueeze(1),

                float("-inf")

            )
 
            # Keep the strongest activation from each filter.

            pooled = features.max(dim=2).values
 
            pooled_features.append(pooled)
 
        # Combine features from kernel sizes 3, 4, and 5.

        combined = torch.cat(pooled_features, dim=1)
 
        # Output three raw scores: GPT, Claude, Gemini.

        return self.fc(self.dropout(combined))
 
 
model = TextCNN(len(vocab))

print(model)
 
 
# 7. Train
 
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001

)
 
loss_fn = nn.CrossEntropyLoss()
 
EPOCHS = 10
 
best_val_accuracy = -1

best_model_state = None

best_epoch = None
 
for epoch in range(EPOCHS):
 
    # Enable dropout during training.

    model.train()
 
    total_loss = 0
 
    for x, lengths, y in train_dl:
 
        # Clear gradients from the previous batch.

        optimizer.zero_grad()
 
        # Predict the family.

        predictions = model(x, lengths)
 
        # Compare predictions against the correct labels.

        loss = loss_fn(predictions, y)
 
        # Calculate gradients.

        loss.backward()
 
        # Limit excessively large gradients.

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
        # Update the model.

        optimizer.step()
 
        total_loss += loss.item() * y.size(0)
 
    average_loss = total_loss / len(train_ds)
 
    # Validation: disable dropout and do not update weights.

    model.eval()
 
    val_predictions = []

    val_labels = []
 
    with torch.no_grad():

        for x, lengths, y in val_dl:
 
            predictions = model(x, lengths)

            predicted_labels = predictions.argmax(dim=1)
 
            val_predictions.extend(predicted_labels.tolist())

            val_labels.extend(y.tolist())
 
    val_accuracy = accuracy_score(

        val_labels,

        val_predictions

    )
 
    print(

        f"Epoch {epoch + 1}/{EPOCHS} "

        f"- Loss: {average_loss:.4f} "

        f"- Val Accuracy: {val_accuracy:.4f}"

    )
 
    # Keep a separate copy of the best weights.

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(model.state_dict())
 
 
# 8. Restore the best model
 
model.load_state_dict(best_model_state)
 
print("\nBest epoch:", best_epoch)

print("Best validation accuracy:", best_val_accuracy)
 
 
# 9. Final test evaluation
 
model.eval()
 
test_predictions = []

test_labels = []
 
with torch.no_grad():

    for x, lengths, y in test_dl:
 
        predictions = model(x, lengths)

        predicted_labels = predictions.argmax(dim=1)
 
        test_predictions.extend(predicted_labels.tolist())

        test_labels.extend(y.tolist())
 
 
# Accuracy

test_accuracy = accuracy_score(

    test_labels,

    test_predictions

)
 
# Macro F1: gives each family equal weight.

test_macro_f1 = f1_score(

    test_labels,

    test_predictions,

    labels=[0, 1, 2],

    average="macro",

    zero_division=0

)
 
# Confusion matrix

cm = confusion_matrix(

    test_labels,

    test_predictions,

    labels=[0, 1, 2]

)
 
print("\n===== RQ1 CNN TEST RESULTS =====")

print("Test Accuracy:", test_accuracy)

print("Test Macro F1:", test_macro_f1)
 
print("\nConfusion Matrix — rows: actual, columns: predicted")

print(pd.DataFrame(

    cm,

    index=family_names,

    columns=family_names

))
 