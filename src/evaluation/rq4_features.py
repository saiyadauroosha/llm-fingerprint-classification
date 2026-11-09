"""Evaluate RQ4 length and formatting features with fixed logistic regression.

Each feature describes only the first 512 response tokens. Formatting uses rates
and flags, which can still correlate with length. Compare length against length
plus formatting to test whether formatting adds predictive information.
By default, use the six supplied tables under data/analysis_features.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime
from io import BytesIO
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ["GPT", "Claude", "Gemini"]
MAX_TOKENS = 512
TOKEN = re.compile(r"\w+|\n|[^\w\s]")
WORD = re.compile(r"\w+")
PUNCTUATION = re.compile(r"[^\w\s]")
HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S", re.MULTILINE)
BULLET = re.compile(r"^[ \t]{0,3}[-+*][ \t]+\S", re.MULTILINE)
NUMBERED = re.compile(r"^[ \t]{0,3}\d+[.)][ \t]+\S", re.MULTILINE)
FENCE = re.compile(r"^[ \t]{0,3}```", re.MULTILINE)
LENGTH = ["token_count", "word_token_count", "nonwhitespace_character_count"]
FORMATTING_RATES = [
    "punctuation_per_100_tokens", "newlines_per_100_tokens",
    "heading_lines_per_100_tokens", "bullet_lines_per_100_tokens",
    "numbered_lines_per_100_tokens",
]
FORMATTING_FLAGS = ["has_bullets", "has_heading"]
SUPPLIED_FORMATTING = FORMATTING_RATES + FORMATTING_FLAGS
FORMATTING = FORMATTING_RATES + ["code_fence_lines_per_100_tokens"] + FORMATTING_FLAGS
SPLITS = ("train", "validation", "test")
MODEL_SETTINGS = {"solver": "lbfgs", "C": 1.0, "max_iter": 1000, "random_state": 42}


def response_features(response):
    text = response.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(islice(TOKEN.finditer(text), MAX_TOKENS))
    tokens = [match.group() for match in matches]
    prefix = text[:matches[-1].end()] if matches else ""
    rate = 100 / max(1, len(tokens))
    headings = len(HEADING.findall(prefix))
    bullets = len(BULLET.findall(prefix))
    return [
        len(tokens),
        sum(WORD.fullmatch(token) is not None for token in tokens),
        sum(len(token) for token in tokens if token != "\n"),
        rate * sum(PUNCTUATION.fullmatch(token) is not None for token in tokens),
        rate * tokens.count("\n"),
        rate * headings,
        rate * bullets,
        rate * len(NUMBERED.findall(prefix)),
        rate * len(FENCE.findall(prefix)),
        int(bullets > 0),
        int(headings > 0),
    ]


def load_supplied_features(feature_dir, frames):
    """Read the supplied CSV values and align them to the text baseline by ID."""
    features, hashes = {}, {}
    for split, frame in frames.items():
        expected = frame.set_index("sample_id").family
        tables = []
        for name, columns in (("length_only", LENGTH), ("formatting_only", SUPPLIED_FORMATTING)):
            path = feature_dir / f"{name}_{split}.csv"
            data = path.read_bytes()
            table = pd.read_csv(BytesIO(data), usecols=["sample_id", "family", "split", *columns])
            table = table.set_index("sample_id")
            if not table.index.is_unique or set(table.index) != set(expected.index):
                raise ValueError(f"{path}: sample IDs must match the selected text split exactly.")
            table = table.loc[expected.index]
            if not table.family.eq(expected).all() or not table.split.eq(split).all():
                raise ValueError(f"{path}: family labels or split assignments differ.")
            numeric = table[columns].apply(pd.to_numeric)
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError(f"{path}: feature values must be finite numbers.")
            tables.append(numeric.reset_index(drop=True))
            hashes[path.name] = hashlib.sha256(data).hexdigest()
        features[split] = pd.concat(tables, axis=1)
    return features, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rq4_features")
    feature_input = parser.add_mutually_exclusive_group()
    feature_input.add_argument("--feature-dir", type=Path, default=ROOT / "data" / "analysis_features",
                               help="Directory containing the six supplied feature CSVs.")
    feature_input.add_argument("--regenerate", action="store_true",
                               help="Compute features from the selected Parquet responses.")
    args = parser.parse_args()

    frames, features, hashes = {}, {}, {}
    seen_ids = set()
    columns = ["sample_id", "family"]
    if args.regenerate:
        columns.append("response")
    for split in SPLITS:
        data = (args.data_dir / f"{split}.parquet").read_bytes()
        frame = pd.read_parquet(BytesIO(data), columns=columns)
        if frame.empty or frame.isna().any().any() or not frame.family.isin(FAMILIES).all():
            raise ValueError(f"{split}: missing rows, values, or valid family labels.")
        if not frame.sample_id.is_unique or seen_ids.intersection(frame.sample_id):
            raise ValueError(f"{split}: sample IDs must be unique across all splits.")
        seen_ids.update(frame.sample_id)
        frames[split] = frame
        hashes[split] = hashlib.sha256(data).hexdigest()
    if set(frames["train"].family) != set(FAMILIES):
        raise ValueError("Training requires all three families.")

    if args.regenerate:
        formatting = FORMATTING
        source = {"kind": "regenerated_from_responses"}
        features = {
            split: pd.DataFrame([response_features(response) for response in frame.response],
                                columns=LENGTH + formatting)
            for split, frame in frames.items()
        }
    else:
        formatting = SUPPLIED_FORMATTING
        features, feature_hashes = load_supplied_features(args.feature_dir, frames)
        source = {"kind": "supplied_csv_files", "sha256": feature_hashes}

    output = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    if args.regenerate:
        for split, frame in frames.items():
            pd.concat([frame[["sample_id", "family"]].reset_index(drop=True), features[split]], axis=1).to_csv(
                output / f"{split}_features.csv", index=False
            )

    results = {}
    test_predictions = frames["test"][["sample_id", "family"]].copy()
    for name, columns in (("length", LENGTH), ("formatting", formatting),
                          ("length_plus_formatting", LENGTH + formatting)):
        model = make_pipeline(StandardScaler(), LogisticRegression(**MODEL_SETTINGS))
        model.fit(features["train"][columns], frames["train"].family)
        scores = {}
        for split in ("validation", "test"):
            predictions = model.predict(features[split][columns])
            labels = frames[split].family
            scores[split] = {
                "accuracy": accuracy_score(labels, predictions),
                "macro_f1": f1_score(labels, predictions, labels=FAMILIES,
                                     average="macro", zero_division=0),
            }
            if split == "test":
                test_predictions[f"{name}_prediction"] = predictions
        results[name] = {"features": columns, **scores}
        print(f"{name}: {scores['test']['accuracy']:.4f} accuracy, "
              f"{scores['test']['macro_f1']:.4f} macro F1")

    test_predictions.to_csv(output / "test_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps({
        "max_tokens": MAX_TOKENS,
        "feature_source": source,
        "split_sha256": hashes,
        "split_counts": {split: len(frame) for split, frame in frames.items()},
        "logistic_regression": MODEL_SETTINGS,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
