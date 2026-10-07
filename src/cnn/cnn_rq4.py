"""RQ4: Which writing characteristics and local patterns distinguish the families?

Self-contained: imports only installed libraries, never other project Python files.
Run from the repository root:
    python src/cnn/rq4_cnn.py

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
MAX_LEN = 512                 # Response token budget in every experiment
PROMPT_LEN = 128              # Additional prompt budget for RQ2
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
    counts = Counter(token for text in train.response for token in tokenize(text))
    if mode == "prompt_response":
        counts.update(token for text in train.prompt for token in tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
    for token, count in counts.items():
        if count >= MIN_FREQUENCY and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


# 4. Convert one example into token IDs and record its true length.
def text_tokens(row, mode="response"):
    response = tokenize(row["response"])[:MAX_LEN]
    if mode == "response":
        return response
    if mode == "prompt_response":
        # Adding the prompt never reduces the number of retained response tokens.
        return tokenize(row["prompt"])[:PROMPT_LEN] + ["<sep>"] + response
    raise ValueError(f"Unknown input mode: {mode}")


def encode(row, vocab, mode="response"):
    tokens = text_tokens(row, mode)
    ids = [vocab.get(token, 1) for token in tokens] or [1]
    length = len(ids)
    width = MAX_LEN if mode == "response" else PROMPT_LEN + 1 + MAX_LEN
    width = max(width, max(KERNEL_SIZES))
    ids += [0] * (width - length)
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
                  "label2id": label2id, "mode": mode, "max_len": MAX_LEN,
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


# 10. RQ4: measure writing style and inspect patterns supporting CNN predictions.
def writing_style_summary(train):
    # Descriptive statistics use FULL training responses. The CNN itself sees
    # only the first MAX_LEN tokens, so these are not direct feature importances.
    rows = []
    for row in train.to_dict("records"):
        text = row["response"]
        words = re.findall(r"\w+", text)
        letters = [character for character in text if character.isalpha()]
        rows.append({"family": row["family"], "word_count": len(words),
                     "mean_word_length": sum(map(len, words)) / max(len(words), 1),
                     "newlines_per_100_words": 100 * text.count("\n") / max(len(words), 1),
                     "punctuation_per_100_chars": 100 * len(re.findall(r"[^\w\s]", text)) / max(len(text), 1),
                     "uppercase_letter_fraction": sum(c.isupper() for c in letters) / max(len(letters), 1),
                     "has_bullet_list": int(bool(re.search(r"(?m)^\s*(?:[-*+] |\d+[.)] )", text))),
                     "has_markdown_heading": int(bool(re.search(r"(?m)^\s*#{1,6}\s+", text))),
                     "has_code_fence": int("```" in text)})
    values = pd.DataFrame(rows)
    summary = values.groupby("family").mean().reindex(CLASS_NAMES)
    summary.insert(0, "n_responses", train.family.value_counts().reindex(CLASS_NAMES))
    return summary


def cnn_supporting_patterns(model, val, vocab, device, top_n=20, min_examples=2):
    # Interpret VALIDATION examples, not the final test examples.
    # In eval mode the head is linear: class_logit = bias + sum(weight * pooled_activation).
    # Subtract the average other-class weight to explain a class-vs-other-classes
    # logit margin. The bias is constant and is not attributed to a text pattern.
    model.eval()
    loader = make_loader(val, vocab)
    original_tokens = [text_tokens(row) for row in val.to_dict("records")]
    totals = defaultdict(lambda: {"total_margin_contribution": 0.0, "example_count": 0})
    analyzed = Counter()
    offset = 0
    filters_per_kernel = model.convs[0].out_channels
    with torch.no_grad():
        for x, lengths, y in loader:
            features, positions = model.extract_features(x.to(device), lengths.to(device))
            predictions = model.fc(features).argmax(dim=1)
            weights = model.fc.weight
            for row_index, true_id in enumerate(y.tolist()):
                if predictions[row_index].item() != true_id:
                    continue   # Explain successful classifications, stated in the report.
                family = CLASS_NAMES[true_id]
                analyzed[family] += 1
                other_ids = [i for i in CLASS_IDS if i != true_id]
                margin_weights = weights[true_id] - weights[other_ids].mean(dim=0)
                contributions = (features[row_index] * margin_weights).cpu().tolist()
                # Sum a repeated pattern within one example, but count that example once.
                per_example = defaultdict(float)
                tokens = original_tokens[offset + row_index]
                for feature_index, contribution in enumerate(contributions):
                    if contribution <= 0:
                        continue
                    kernel = model.kernel_sizes[feature_index // filters_per_kernel]
                    start = int(positions[row_index, feature_index].item())
                    window = tokens[start:start + kernel]
                    if len(window) < kernel or any(token not in vocab for token in window):
                        continue  # Do not present unseen words as words the model recognized.
                    pattern = " ".join("<NEWLINE>" if token == "\n" else token for token in window)
                    per_example[(family, kernel, pattern)] += contribution
                for key, contribution in per_example.items():
                    totals[key]["total_margin_contribution"] += contribution
                    totals[key]["example_count"] += 1
            offset += len(y)

    rows = []
    for (family, kernel, pattern), values in totals.items():
        if values["example_count"] >= min_examples:
            rows.append({"family": family, "kernel_size": kernel, "pattern": pattern, **values,
                         "mean_margin_contribution": values["total_margin_contribution"] / values["example_count"]})
    columns = ["family", "kernel_size", "pattern", "total_margin_contribution", "example_count", "mean_margin_contribution"]
    table = pd.DataFrame(rows, columns=columns)
    if not table.empty:
        table = table.sort_values(["family", "total_margin_contribution"], ascending=[True, False])
        table = table.groupby("family", sort=False).head(top_n).reset_index(drop=True)
    return table, {family: analyzed[family] for family in CLASS_NAMES}


def main(argv=None):
    def extra(parser):
        parser.add_argument("--top-patterns", type=int, default=20)
        parser.add_argument("--min-pattern-examples", type=int, default=2)
    args, device, output = setup("rq4", argv, extra)
    if args.top_patterns < 1 or args.min_pattern_examples < 1:
        raise ValueError("Pattern limits must be positive.")
    train, val, test = load_data(args.data_dir)
    model, vocab, checkpoint, history = train_cnn(train, val, "response", device, args.epochs)
    folder = output / "response"
    save_model(folder, checkpoint, history)

    style = writing_style_summary(train)
    style.to_csv(output / "rq4_training_style_by_family.csv")
    patterns, analyzed = cnn_supporting_patterns(model, val, vocab, device,
                                                args.top_patterns, args.min_pattern_examples)
    patterns.to_csv(output / "rq4_validation_supporting_patterns.csv", index=False)
    (output / "rq4_interpretation_notes.json").write_text(json.dumps({
        "style_data": "Full training responses; descriptive means, not CNN feature importance.",
        "pattern_data": "Correctly classified validation responses, truncated to MAX_LEN.",
        "correct_validation_examples_analyzed": analyzed,
        "pattern_score": "Positive pooled-feature contributions to true-class logit minus mean other-class logits; bias excluded.",
        "ranking": "Total positive contribution, restricted to patterns meeting minimum example count.",
        "minimum_example_count": args.min_pattern_examples,
        "limitations": "Associations in this dataset and model. They may reflect topic, length or model version, not universal authorship rules. Counts are restricted to correctly classified validation examples."
    }, indent=2) + "\n")

    # Test performance is reported once; test examples never select the patterns.
    test_and_save(model, test, vocab, "response", device, folder)
    print("\n===== RQ4: MEAN WRITING-STYLE STATISTICS =====\n", style.round(3).to_string())
    print("\n===== RQ4: CNN-SUPPORTING VALIDATION PATTERNS =====\n", patterns.to_string(index=False))
    if patterns.empty:
        print("No patterns met the minimum support requirement; an empty results file was saved.")
    print("These are descriptive/model associations, not proof that a pattern uniquely identifies an LLM.")
    return style, patterns


if __name__ == "__main__":
    main()
