"""RQ2: Does the prompt help a CNN identify the LLM family?

Compares three inputs, using the same format as the RNN team (src/rnn/rnn_rq2.py):
    prompt           -> first 128 prompt tokens
    response         -> first 512 response tokens
    prompt_response  -> first 128 prompt tokens + <sep> + first 512 response tokens

Self-contained: imports only installed libraries, never other project Python files.
Run from the repository root:
    python src/cnn/cnn_rq2.py

Optional: --data-dir PATH --output-dir NEW_PATH --epochs 10 --device cpu
Edit the settings below to change embedding size, filters or input lengths.
Uses the existing fixed splits and a training-only vocabulary.
Model selection uses validation accuracy, matching the supplied LSTM example.
Architecture reference: https://aclanthology.org/D14-1181/
"""

import argparse
import json
import random
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# 1. Settings — edit these values to change the experiment.
# These match the basic setup of your LSTM example where applicable.
MAX_LEN = 512                 # Response token budget (response, prompt_response)
PROMPT_LEN = 128              # Prompt token budget (prompt, prompt_response)
MODES = ("prompt", "response", "prompt_response")
EMB_DIM = 200
NUM_FILTERS = 100             # Number of filters for EACH kernel size
KERNEL_SIZES = (3, 4, 5)
DROPOUT = 0.5
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
MIN_FREQUENCY = 2
SEED = 42

label2id = {"GPT": 0, "Claude": 1, "Gemini": 2}
CLASS_NAMES = list(label2id)
CLASS_IDS = list(label2id.values())


# 2. Load the existing splits. Do not randomly split the dataset again.
def load_data(data_dir):
    frames = {}
    required = ["sample_id", "prompt_group_id", "family", "prompt", "response"]
    for split in ("train", "validation", "test"):
        frame = pd.read_parquet(Path(data_dir) / f"{split}.parquet").reset_index(drop=True)
        if set(required) - set(frame.columns):
            raise ValueError(f"{split}: missing required columns.")
        if frame.empty or frame[required].isna().any().any():
            raise ValueError(f"{split}: empty data or missing required values.")
        if frame.sample_id.duplicated().any():
            raise ValueError(f"{split}: sample IDs must be unique.")
        for column in ("family", "prompt", "response"):
            if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
                raise ValueError(f"{split}: {column} must contain nonempty text.")
        if set(frame.family) != set(CLASS_NAMES):
            raise ValueError(f"{split}: expected GPT, Claude and Gemini.")
        if "split" in frame and not frame["split"].eq(split).all():
            raise ValueError(f"{split}: the split column does not match the file.")
        frames[split] = frame

    for first, second in combinations(frames, 2):
        for key in ("sample_id", "prompt_group_id"):
            if set(frames[first][key]) & set(frames[second][key]):
                raise ValueError(f"{key} overlaps between {first} and {second}.")
        normalize = lambda text: " ".join(text.split()).casefold()
        prompts = set(frames[first].prompt.map(normalize)) & set(frames[second].prompt.map(normalize))
        responses = set(frames[first].response) & set(frames[second].response)
        if prompts or responses:
            warnings.warn(f"{first}/{second}: {len(prompts)} shared normalized prompts and "
                          f"{len(responses)} shared exact response texts. Fixed splits preserved.")
    return frames["train"], frames["validation"], frames["test"]


# 3. Tokenize and build the vocabulary from TRAINING text only.
def tokenize(text):
    # Preserve case, newlines, punctuation and Markdown symbols.
    # Spaces and tabs are not separate tokens.
    return re.findall(r"\w+|\n|[^\w\s]", text.replace("\r\n", "\n").replace("\r", "\n"))


def build_vocab(train, mode="response"):
    # Count only the tokens the model actually sees for this input mode,
    # matching the RNN team's per-input vocabulary.
    counts = Counter(token for row in train.to_dict("records")
                     for token in text_tokens(row, mode) if token != "<sep>")
    vocab = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
    for token, count in counts.items():
        if count >= MIN_FREQUENCY and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


# 4. Convert one example into token IDs and record its true length.
def text_tokens(row, mode="response"):
    if mode == "prompt":
        return tokenize(row["prompt"])[:PROMPT_LEN]
    if mode == "response":
        return tokenize(row["response"])[:MAX_LEN]
    if mode == "prompt_response":
        # <sep> marks "prompt ends here, response starts here". Adding the
        # prompt never reduces the number of retained response tokens.
        return (tokenize(row["prompt"])[:PROMPT_LEN] + ["<sep>"]
                + tokenize(row["response"])[:MAX_LEN])
    raise ValueError(f"Unknown input mode: {mode}")


def input_width(mode):
    # 128 for prompt, 512 for response, 128 + 1 + 512 = 641 for prompt_response.
    width = {"prompt": PROMPT_LEN, "response": MAX_LEN,
             "prompt_response": PROMPT_LEN + 1 + MAX_LEN}[mode]
    return max(width, max(KERNEL_SIZES))


def encode(row, vocab, mode="response"):
    tokens = text_tokens(row, mode)
    ids = [vocab.get(token, 1) for token in tokens] or [1]
    length = len(ids)
    ids += [0] * (input_width(mode) - length)
    return ids, length


# 5. Dataset and batches.
class TextDataset(Dataset):
    def __init__(self, frame, vocab, mode="response"):
        encoded = [encode(row, vocab, mode) for row in frame.to_dict("records")]
        self.x = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        self.lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)
        self.y = torch.tensor([label2id[family] for family in frame.family], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.lengths[index], self.y[index]


def make_loader(frame, vocab, mode="response", shuffle=False):
    if frame.empty:
        raise ValueError("Cannot train or evaluate an empty subset.")
    return DataLoader(TextDataset(frame, vocab, mode), batch_size=BATCH_SIZE,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(SEED))


# 6. CNN: embedding -> parallel convolutions -> max pooling -> dropout -> linear.
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMB_DIM, num_filters=NUM_FILTERS,
                 kernel_sizes=KERNEL_SIZES, dropout=DROPOUT):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, kernel_size=k) for k in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), len(CLASS_NAMES))

    def extract_features(self, x, lengths):
        # [batch, length, embedding] -> [batch, embedding, length]
        if x.size(1) < max(self.kernel_sizes):
            x = nn.functional.pad(x, (0, max(self.kernel_sizes) - x.size(1)))
        embedded = self.embedding(x).transpose(1, 2)
        pooled, locations = [], []
        for kernel_size, convolution in zip(self.kernel_sizes, self.convs):
            activations = torch.relu(convolution(embedded))
            # Ignore windows touching right-padding. Very short texts get one
            # zero-padded window, so a 1-word response can still be classified.
            valid_windows = (lengths - kernel_size + 1).clamp(min=1)
            positions = torch.arange(activations.size(2), device=x.device)
            padding_mask = positions[None, :] >= valid_windows[:, None]
            activations = activations.masked_fill(padding_mask[:, None, :], float("-inf"))
            values, starts = activations.max(dim=2)
            pooled.append(values)
            locations.append(starts)
        # 100 filters x 3 kernel sizes = 300 features with the defaults.
        return torch.cat(pooled, dim=1), torch.cat(locations, dim=1)

    def forward(self, x, lengths):
        features, _ = self.extract_features(x, lengths)
        # Return logits. CrossEntropyLoss handles the required normalization.
        return self.fc(self.dropout(features))


def evaluate(model, loader, device):
    model.eval()              # Disable dropout.
    actual, predicted = [], []
    with torch.no_grad():     # Evaluation does not update weights.
        for x, lengths, y in loader:
            logits = model(x.to(device), lengths.to(device))
            actual.extend(y.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    scores = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=CLASS_IDS,
                                    average="macro", zero_division=0)),
    }
    return scores, actual, predicted


# 7. Train and copy the best weights using VALIDATION accuracy.
# This uses the same selection metric as your pasted LSTM example.
def train_cnn(train, val, mode, device, epochs):
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for name, frame in (("training", train), ("validation", val)):
        if set(frame.family) != set(CLASS_NAMES):
            raise ValueError(f"All three families must occur in the {name} subset.")
    vocab = build_vocab(train, mode)
    train_dl = make_loader(train, vocab, mode, shuffle=True)
    val_dl = make_loader(val, vocab, mode)
    model_args = dict(vocab_size=len(vocab), emb_dim=EMB_DIM, num_filters=NUM_FILTERS,
                      kernel_sizes=list(KERNEL_SIZES), dropout=DROPOUT)
    model = TextCNN(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy, best_state, best_epoch = -1.0, None, None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()         # Enable dropout during training.
        total_loss = 0.0
        for x, lengths, y in train_dl:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        scores, _, _ = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_dl.dataset),
                        "val_accuracy": scores["accuracy"], "val_macro_f1": scores["macro_f1"]})
        print(f"Epoch {epoch}/{epochs} | Loss {history[-1]['train_loss']:.4f} | "
              f"Val accuracy {scores['accuracy']:.4f} | Val F1 {scores['macro_f1']:.4f}", flush=True)
        if scores["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = scores["accuracy"], epoch
            # state_dict() alone points to live weights. Clone each tensor so
            # later epochs cannot change this saved best-model snapshot.
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    checkpoint = {"model_state": best_state, "model_args": model_args, "vocab": vocab,
                  "label2id": label2id, "mode": mode, "input_width": input_width(mode),
                  "max_len": MAX_LEN,
                  "prompt_len": PROMPT_LEN, "best_epoch": best_epoch,
                  "selection_metric": "validation_accuracy", "best_val_accuracy": best_accuracy,
                  "seed": SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                  "min_frequency": MIN_FREQUENCY, "epochs": epochs,
                  "training_sample_ids": train.sample_id.astype(str).tolist(),
                  "validation_sample_ids": val.sample_id.astype(str).tolist()}
    return model, vocab, checkpoint, history


# 8. Save the restored best model, test scores, and a labeled confusion matrix.
def save_model(folder, checkpoint, history):
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, folder / "best_model.pt")
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)


def test_and_save(model, frame, vocab, mode, device, folder, name="test"):
    scores, actual, predicted = evaluate(model, make_loader(frame, vocab, mode), device)
    matrix = confusion_matrix(actual, predicted, labels=CLASS_IDS)
    report = classification_report(actual, predicted, labels=CLASS_IDS,
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    (folder / f"{name}_metrics.json").write_text(
        json.dumps({**scores, "n_examples": len(frame), "classification_report": report}, indent=2) + "\n")
    pd.DataFrame(matrix, index=pd.Index(CLASS_NAMES, name="actual"),
                 columns=CLASS_NAMES).to_csv(folder / f"{name}_confusion_matrix.csv")
    predictions = frame[["sample_id", "family"]].copy()
    predictions["predicted_family"] = [CLASS_NAMES[index] for index in predicted]
    predictions.to_csv(folder / f"{name}_predictions.csv", index=False)
    print(f"{name}: accuracy={scores['accuracy']:.4f}, macro F1={scores['macro_f1']:.4f}", flush=True)
    print(pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES), flush=True)
    return scores


# 9. Locate the project and choose an output folder for this run.
def setup(question, argv=None, extra_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    root = next((p for p in candidates if (p / "data/splits/train.parquet").exists()), Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=root / "data/splits")
    parser.add_argument("--output-dir", type=Path, help="A NEW directory for this run")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    if extra_arguments:
        extra_arguments(parser)
    args = parser.parse_args(argv)
    if args.epochs < 1 or min(MAX_LEN, PROMPT_LEN, BATCH_SIZE, EMB_DIM, NUM_FILTERS, MIN_FREQUENCY) < 1:
        parser.error("Epochs, lengths, batch size, dimensions and frequency must be positive.")
    if not KERNEL_SIZES or min(KERNEL_SIZES) < 1 or not 0 <= DROPOUT < 1 or LEARNING_RATE <= 0:
        parser.error("Invalid kernel size, dropout, or learning rate.")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir or root / "results/cnn" / question / timestamp
    # Preserve previous runs instead of overwriting them.
    output.mkdir(parents=True, exist_ok=False)
    print(f"Device: {device}\nResults: {output}", flush=True)
    return args, torch.device(device), output


# 10. RQ2: compare prompt only, response only, and prompt + response.
def main(argv=None):
    args, device, output = setup("rq2", argv)
    train, val, test = load_data(args.data_dir)
    comparison = []
    for mode in MODES:
        print(f"\n===== TRAINING: {mode} =====")
        # Reset the same seed and use the same split files, response budget,
        # embedding size, filters, optimizer, and epoch count for every model.
        model, vocab, checkpoint, history = train_cnn(train, val, mode, device, args.epochs)
        folder = output / mode
        save_model(folder, checkpoint, history)
        scores = test_and_save(model, test, vocab, mode, device, folder)
        comparison.append({"input": mode, "input_tokens": input_width(mode),
                           "best_epoch": checkpoint["best_epoch"],
                           "vocab_size": len(vocab), **scores})
        del model

    table = pd.DataFrame(comparison)
    table.to_csv(output / "rq2_comparison.csv", index=False)
    print("\n===== RQ2 COMPARISON =====\n", table.to_string(index=False))
    results = {row["input"]: row for row in comparison}
    for label, first, second in (("Prompt only vs response only", "response", "prompt"),
                                 ("Adding the prompt to the response", "response", "prompt_response")):
        accuracy_change = results[second]["accuracy"] - results[first]["accuracy"]
        f1_change = results[second]["macro_f1"] - results[first]["macro_f1"]
        print(f"{label}: accuracy {100 * accuracy_change:+.2f} percentage points, "
              f"macro F1 {f1_change:+.4f}")
    print("This is a single-seed comparison. Repeat predefined seeds before claiming a reliable improvement.")
    print("Prompt inclusion also changes the vocabulary; vocabulary sizes are reported above.")
    return table


if __name__ == "__main__":
    main()
