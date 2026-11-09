# RQ1: Identify the LLM from its response using a CNN
 
import argparse
import copy
import hashlib
import json
import platform
import re

from collections import Counter
from datetime import datetime
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
 
import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
 
 
# 1. Load the fixed train/validation/test splits

# Resolve paths from this file, so the working directory does not matter.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAX_LEN = 512
MIN_FREQUENCY = 2
SEED = 42

parser = argparse.ArgumentParser(description="Train the RQ1 response-only CNN.")
parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "splits",
                    help="Directory containing train, validation, and test Parquet files.")
parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "cnn" / "rq1",
                    help="Parent directory for the timestamped run folder.")
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()
if args.epochs < 1:
    parser.error("--epochs must be at least 1.")

frames, split_sha256 = {}, {}
for split in ("train", "validation", "test"):
    split_bytes = (args.data_dir / f"{split}.parquet").read_bytes()
    frames[split] = pd.read_parquet(BytesIO(split_bytes))
    if frames[split].empty:
        raise ValueError(f"{split}: cannot run on an empty split.")
    split_sha256[split] = hashlib.sha256(split_bytes).hexdigest()
train, val, test = frames["train"], frames["validation"], frames["test"]

# Keep each run in a new folder so earlier results are not overwritten.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUTPUT_DIR = args.output_dir / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
print("Results folder:", OUTPUT_DIR, flush=True)

torch.manual_seed(SEED)
 
# Convert family names into integer labels.

label2id = {

    "GPT": 0,

    "Claude": 1,

    "Gemini": 2

}
 
family_names = list(label2id.keys())
 
 
# 2. Tokenize
 
def tokenize(text):

    # Match RQ2's treatment of line endings and keep structural tokens.
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return re.findall(r"\w+|\n|[^\w\s]", text)[:MAX_LEN]
 
 
# 3. Count only the training tokens within the model's response window.
 
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

    if count >= MIN_FREQUENCY:

        vocab[token] = len(vocab)
 
print("Vocabulary size:", len(vocab))
 
 
# 4. Convert text into numbers
 
def encode(text):

    tokens = tokenize(text)
 
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

    shuffle=True,
    generator=torch.Generator().manual_seed(SEED)

)
 
val_dl = DataLoader(

    val_ds,

    batch_size=64,

    shuffle=False,
    generator=torch.Generator().manual_seed(SEED)

)
 
test_dl = DataLoader(

    test_ds,

    batch_size=64,

    shuffle=False,
    generator=torch.Generator().manual_seed(SEED)

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
 
EPOCHS = args.epochs
 
best_val_accuracy = -1

best_model_state = None

best_epoch = None

history = []
 
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

    val_macro_f1 = f1_score(
        val_labels,
        val_predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": average_loss,
        "val_accuracy": float(val_accuracy),
        "val_macro_f1": float(val_macro_f1)
    })
 
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

model_args = {
    "vocab_size": len(vocab),
    "emb_dim": model.embedding.embedding_dim,
    "num_filters": model.convs[0].out_channels,
    "kernel_sizes": list(model.kernel_sizes),
    "n_classes": model.fc.out_features,
    "dropout": model.dropout.p
}
training = {
    "mode": "response",
    "max_len": MAX_LEN,
    "vocab_source": "training_response_prefix",
    "best_epoch": best_epoch,
    "selection_metric": "validation_accuracy",
    "best_val_accuracy": float(best_val_accuracy),
    "seed": SEED,
    "batch_size": train_dl.batch_size,
    "learning_rate": optimizer.param_groups[0]["lr"],
    "min_frequency": MIN_FREQUENCY,
    "epochs": EPOCHS
}
checkpoint = {
    "model_state": best_model_state,
    "model_args": model_args,
    "vocab": vocab,
    "label2id": label2id,
    **training,
    "training_sample_ids": train.sample_id.astype(str).tolist(),
    "validation_sample_ids": val.sample_id.astype(str).tolist()
}
torch.save(checkpoint, OUTPUT_DIR / "best_model.pt")
pd.DataFrame(history).to_csv(OUTPUT_DIR / "history.csv", index=False)

metadata = {
    "experiment": "RQ1 CNN response-only",
    "script": "src/cnn/cnn_rq1.py",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "device": str(next(model.parameters()).device),
    "packages": {
        "python": platform.python_version(),
        **{package: version(package) for package in ("torch", "numpy", "pandas", "pyarrow")},
        "scikit_learn": version("scikit-learn")
    },
    "model_args": model_args,
    "training": training,
    "split_counts": {name: len(frame) for name, frame in frames.items()},
    "split_sha256": split_sha256
}
(OUTPUT_DIR / "run_metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
)
 
 
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

# 10. Save final test scores, the confusion matrix, and individual predictions.
report = classification_report(
    test_labels,
    test_predictions,
    labels=[0, 1, 2],
    target_names=family_names,
    output_dict=True,
    zero_division=0
)
metrics = {
    "accuracy": float(test_accuracy),
    "macro_precision": report["macro avg"]["precision"],
    "macro_recall": report["macro avg"]["recall"],
    "macro_f1": float(test_macro_f1),
    "n_examples": len(test),
    "classification_report": report
}
(OUTPUT_DIR / "test_metrics.json").write_text(
    json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
)
pd.DataFrame(
    cm,
    index=pd.Index(family_names, name="actual"),
    columns=family_names
).to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")

# test_dl uses shuffle=False, so predictions follow the original test-row order.
predictions = test[["sample_id", "family"]].copy()
predictions["predicted_family"] = [family_names[index] for index in test_predictions]
predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

print("\nSaved model and results to:", OUTPUT_DIR)
