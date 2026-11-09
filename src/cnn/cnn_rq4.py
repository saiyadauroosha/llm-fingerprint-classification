"""RQ4: Which writing characteristics and local patterns distinguish the families?

Self-contained: imports only installed libraries, never other project Python files.
Run from the repository root:
    python src/cnn/cnn_rq4.py --checkpoint PATH_TO_RQ1_BEST_MODEL.pt

Optional: --data-dir PATH --output-dir NEW_PATH --device cpu
Keep the RQ1 run_metadata.json beside the checkpoint. Pass the same data
directory used for RQ1 so the split fingerprints match.
Architecture reference: https://aclanthology.org/D14-1181/
"""

import argparse
import hashlib
import json
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


label2id = {"GPT": 0, "Claude": 1, "Gemini": 2}
CLASS_NAMES = list(label2id)
CLASS_IDS = list(label2id.values())


# Load the exact splits from the baseline run.
def load_data(data_dir, split_sha256):
    frames = {}
    required = ["sample_id", "prompt_group_id", "family", "prompt", "response"]
    for split in ("train", "validation", "test"):
        split_bytes = (data_dir / f"{split}.parquet").read_bytes()
        if hashlib.sha256(split_bytes).hexdigest() != split_sha256.get(split):
            raise ValueError(f"{split}: split SHA-256 does not match the RQ1 run metadata.")
        frame = pd.read_parquet(BytesIO(split_bytes)).reset_index(drop=True)
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


# Use the baseline's response tokenizer, input length, and saved vocabulary.
def tokenize(text, max_len):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return re.findall(r"\w+|\n|[^\w\s]", text)[:max_len]


def encode(text, vocab, max_len):
    tokens = tokenize(text, max_len)
    ids = [vocab.get(token, 1) for token in tokens] or [1]
    length = len(ids)
    ids += [0] * (max_len - length)
    return ids, length


# 5. Dataset and batches.
class TextDataset(Dataset):
    def __init__(self, frame, vocab, max_len):
        encoded = [encode(text, vocab, max_len) for text in frame.response]
        self.x = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        self.lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)
        self.y = torch.tensor([label2id[family] for family in frame.family], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.lengths[index], self.y[index]


def make_loader(frame, vocab, max_len, batch_size):
    return DataLoader(TextDataset(frame, vocab, max_len), batch_size=batch_size)


# 6. CNN: embedding -> parallel convolutions -> max pooling -> dropout -> linear.
class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_filters, kernel_sizes, n_classes, dropout):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, kernel_size=k) for k in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), n_classes)

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


def load_rq1_checkpoint(path):
    metadata_path = path.with_name("run_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("script") != "src/cnn/cnn_rq1.py"
        or not metadata.get("split_sha256")
        or not metadata.get("training")
    ):
        raise ValueError(f"{metadata_path}: expected RQ1 CNN run metadata with split hashes.")

    checkpoint_bytes = path.read_bytes()
    checkpoint = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if (
        checkpoint.get("mode") != "response"
        or checkpoint.get("vocab_source") != "training_response_prefix"
        or checkpoint.get("label2id") != label2id
        or checkpoint.get("model_args") != metadata.get("model_args")
        or any(checkpoint.get(key) != value for key, value in metadata["training"].items())
    ):
        raise ValueError(f"{path}: expected a response-only RQ1 CNN checkpoint matching its metadata.")

    model = TextCNN(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint, metadata, hashlib.sha256(checkpoint_bytes).hexdigest()


def test_and_save(model, frame, loader, device, folder):
    scores, actual, predicted = evaluate(model, loader, device)
    matrix = confusion_matrix(actual, predicted, labels=CLASS_IDS)
    report = classification_report(actual, predicted, labels=CLASS_IDS,
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics = {**scores, "macro_precision": report["macro avg"]["precision"],
               "macro_recall": report["macro avg"]["recall"],
               "n_examples": len(frame), "classification_report": report}
    (folder / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(matrix, index=pd.Index(CLASS_NAMES, name="actual"),
                 columns=CLASS_NAMES).to_csv(folder / "test_confusion_matrix.csv")
    predictions = frame[["sample_id", "family"]].copy()
    predictions["predicted_family"] = [CLASS_NAMES[index] for index in predicted]
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    print(f"test: accuracy={scores['accuracy']:.4f}, macro F1={scores['macro_f1']:.4f}", flush=True)
    print(pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES), flush=True)
    return scores


# Locate the project and choose an output folder for this analysis.
def setup(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="RQ1 CNN best_model.pt with run_metadata.json beside it")
    parser.add_argument("--data-dir", type=Path, default=root / "data/splits")
    parser.add_argument("--output-dir", type=Path, help="A NEW directory for this run")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--top-patterns", type=int, default=20)
    parser.add_argument("--min-pattern-examples", type=int, default=2)
    args = parser.parse_args(argv)
    if args.top_patterns < 1 or args.min_pattern_examples < 1:
        parser.error("Pattern limits must be positive.")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir or root / "results/cnn/rq4" / timestamp
    return args, torch.device(device), output


# 10. RQ4: measure writing style and inspect patterns supporting CNN predictions.
def writing_style_summary(train):
    # Descriptive statistics use FULL training responses. The CNN itself sees
    # only the checkpoint's input window.
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


def cnn_supporting_patterns(model, val, loader, vocab, max_len, device, top_n=20, min_examples=2):
    # Interpret VALIDATION examples, not the final test examples.
    # In eval mode the head is linear: class_logit = bias + sum(weight * pooled_activation).
    # Subtract the average other-class weight to explain a class-vs-other-classes
    # logit margin. The bias is constant and is not attributed to a text pattern.
    model.eval()
    original_tokens = [tokenize(text, max_len) for text in val.response]
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
    args, device, output = setup(argv)
    model, checkpoint, metadata, checkpoint_sha256 = load_rq1_checkpoint(args.checkpoint)
    train, val, test = load_data(args.data_dir, metadata["split_sha256"])
    model.to(device)
    vocab = checkpoint["vocab"]
    max_len, batch_size = checkpoint["max_len"], checkpoint["batch_size"]

    folder = output / "response"
    output.mkdir(parents=True, exist_ok=False)
    folder.mkdir()
    print(f"RQ1 checkpoint: {args.checkpoint}\nDevice: {device}\nResults: {output}", flush=True)

    style = writing_style_summary(train)
    style.to_csv(output / "rq4_training_style_by_family.csv")
    val_loader = make_loader(val, vocab, max_len, batch_size)
    patterns, analyzed = cnn_supporting_patterns(
        model, val, val_loader, vocab, max_len, device, args.top_patterns, args.min_pattern_examples
    )
    patterns.to_csv(output / "rq4_validation_supporting_patterns.csv", index=False)
    (output / "rq4_interpretation_notes.json").write_text(json.dumps({
        "baseline_checkpoint": str(args.checkpoint),
        "baseline_checkpoint_sha256": checkpoint_sha256,
        "baseline_best_epoch": checkpoint["best_epoch"],
        "split_sha256": metadata["split_sha256"],
        "style_data": "Full training responses; descriptive means, not CNN feature importance.",
        "pattern_data": f"Correctly classified validation responses, truncated to {max_len} tokens.",
        "correct_validation_examples_analyzed": analyzed,
        "pattern_score": "Positive pooled-feature contributions to true-class logit minus mean other-class logits; bias excluded.",
        "ranking": "Total positive contribution, restricted to patterns meeting minimum example count.",
        "minimum_example_count": args.min_pattern_examples,
        "limitations": "Associations in this dataset and model. They may reflect topic, length or model version, not universal authorship rules. Counts are restricted to correctly classified validation examples."
    }, indent=2) + "\n")

    # Test performance is reported once; test examples never select the patterns.
    test_and_save(model, test, make_loader(test, vocab, max_len, batch_size), device, folder)
    print("\n===== RQ4: MEAN WRITING-STYLE STATISTICS =====\n", style.round(3).to_string())
    print("\n===== RQ4: CNN-SUPPORTING VALIDATION PATTERNS =====\n", patterns.to_string(index=False))
    if patterns.empty:
        print("No patterns met the minimum support requirement; an empty results file was saved.")
    print("These are descriptive/model associations, not proof that a pattern uniquely identifies an LLM.")
    return style, patterns


if __name__ == "__main__":
    main()
