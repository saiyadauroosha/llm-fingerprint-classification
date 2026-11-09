# rnn_rq4.py
#
# RQ4: What characteristics distinguish different LLMs?
#
# Four experiments:
# 1. Full response
# 2. Remove formatting
# 3. Remove punctuation
# 4. First 128 response tokens


import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime
from io import BytesIO
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pack_padded_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "splits"
SEED = 42
MAX_LEN = 512
SHORT_LEN = 128
TOKEN_PATTERN = re.compile(r"\w+|\n|[^\w\s]")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
MARKDOWN_LINE_MARKER = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}(?=[ \t])|(?:[-*+]|\d+[.)])(?=[ \t])"
    r"|>+(?=[ \t]))"
)
MARKDOWN_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


label2id = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}


# 2. Tokenizer

def normalize_line_endings(text):
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def full_response(text):
    return TOKEN_PATTERN.findall(normalize_line_endings(text))[:MAX_LEN]


# 3. Build vocabulary from training data only

def build_vocab(texts):

    counts = Counter(
        token
        for text in texts
        for token in full_response(text)
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

def encode(tokens, vocab):
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens] or [vocab["<unk>"]]
    return ids + [vocab["<pad>"]] * (MAX_LEN - len(ids))


# 5. Dataset

class TextDataset(Dataset):

    def __init__(self, df, vocab, transform):

        self.x = [
            encode(transform(text), vocab)
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



# 7. Response transformations

def remove_formatting(text):
    # Find the same visible prefix as the full-response arm before removing
    # line breaks and Markdown heading, list, quote, and fence markers.
    text = normalize_line_endings(text)
    end = 0
    for match in islice(TOKEN_PATTERN.finditer(text), MAX_LEN):
        end = match.end()
    lines = []
    open_fence = ""
    for line in text[:end].split("\n"):
        match = MARKDOWN_FENCE.match(line)
        if not open_fence:
            if match:
                open_fence = match[1]
                line = line[match.end():]
            else:
                line = MARKDOWN_LINE_MARKER.sub(" ", line)
        elif match and match[1].startswith(open_fence) and not line[match.end():].strip():
            open_fence = ""
            line = line[match.end():]
        lines.append(line)
    visible = "\n".join(lines)
    return [token for token in TOKEN_PATTERN.findall(visible) if token != "\n"]


def remove_punctuation(text):
    # Remove punctuation tokens without collapsing newlines or revealing later tokens.
    return [token for token in full_response(text) if not PUNCTUATION_PATTERN.fullmatch(token)]


def short_response(text):
    return full_response(text)[:SHORT_LEN]


# 8. Run one RQ4 experiment

def run_experiment(train, val, test, vocab, experiment_name, transform, epochs):
    print(f"\n===== {experiment_name} =====", flush=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_dl = DataLoader(
        TextDataset(train, vocab, transform),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_dl = DataLoader(
        TextDataset(val, vocab, transform),
        batch_size=64,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_dl = DataLoader(
        TextDataset(test, vocab, transform),
        batch_size=64,
        generator=torch.Generator().manual_seed(SEED),
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

    best_val_accuracy = -1.0

    best_model = None

    for epoch in range(epochs):

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
        labels=list(label2id.values()),
        average="macro",
        zero_division=0,
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


def main():
    parser = argparse.ArgumentParser(description="Run the four RQ4 LSTM ablations.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "rnn" / "rq4",
                        help="Parent directory for timestamped results")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    frames, split_sha256 = {}, {}
    for name in ("train", "validation", "test"):
        data = (args.data_dir / f"{name}.parquet").read_bytes()
        frames[name] = pd.read_parquet(BytesIO(data))
        split_sha256[name] = hashlib.sha256(data).hexdigest()
    train, val, test = frames["train"], frames["validation"], frames["test"]
    if any(frame.empty for frame in (train, val, test)):
        raise ValueError("Train, validation, and test splits must be nonempty.")

    # Keep the RQ1 vocabulary and token IDs fixed across all ablations.
    vocab = build_vocab(train["response"])
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    experiments = (
        ("full_response", full_response, MAX_LEN),
        ("remove_formatting", remove_formatting, MAX_LEN),
        ("remove_punctuation", remove_punctuation, MAX_LEN),
        ("first_128_tokens", short_response, SHORT_LEN),
    )
    results = []
    for name, transform, max_tokens in experiments:
        accuracy, macro_f1 = run_experiment(train, val, test, vocab, name, transform, args.epochs)
        results.append({
            "experiment": name,
            "max_input_tokens": max_tokens,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "test_examples": len(test),
            "vocab_size": len(vocab),
            "seed": SEED,
            "epochs": args.epochs,
        })
    output = run_dir / "test_metrics.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "split_sha256": split_sha256,
        "split_counts": {name: len(frame) for name, frame in frames.items()},
        "formatting_removal": "Line breaks and Markdown heading, list, quote, and fence markers.",
        "vocab_source": "First 512 tokens from unmodified training responses.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nRQ4 results saved to: {output}")


if __name__ == "__main__":
    main()
