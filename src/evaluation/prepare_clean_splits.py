"""Derive splits_v2 from the audited fixed Arena splits without moving rows."""

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "splits"
OUTPUT = ROOT / "data" / "splits_v2"
SPLITS = ("train", "validation", "test")
SOURCE_HASHES = {
    "train": "47e8ac72b5121b69bb684c704af5c4ca1dbf21ea09bf7e76776609f87b60eaef",
    "validation": "eab64e2e6b4161eb6eaf92ecb0f053f6c2bf1eaa6d3460eb1005aaf2c3cc223b",
    "test": "7230d1d3548d8e8620031c783ff9f20f59a95aeb8ddd1a79fc937519ec32f6be",
}

# Reviewed first-person model or provider disclosures.
IDENTITY_ROWS = {"train": {155, 1130, 1949, 2035, 2753, 5488}, "test": {1979}}
# Later-split sides of reviewed prompt matches. The audit used normalized word
# and punctuation tokens, at least 20 tokens, and 5-token shingle Jaccard >= 0.80.
SIMILAR_PROMPT_ROWS = {
    "validation": {1193, 1892, 2026, 2720, 4864, 5586, 5908},
    "test": {394, 1853, 3997},
}
TOKEN_PATTERN = re.compile(r"\w+|\n|[^\w\s]")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def response_prefix(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return tuple(TOKEN_PATTERN.findall(text)[:512])


def main():
    frames = {}
    for split in SPLITS:
        path = SOURCE / f"{split}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_HASHES[split]:
            raise ValueError(f"{split}: reviewed exclusions apply only to the audited source files.")
        frames[split] = pd.read_parquet(path)

    earlier_prompts, earlier_responses, earlier_prefixes = set(), set(), set()
    cleaned = {}
    exclusions = []
    for split, frame in frames.items():
        prompts = frame.prompt.map(normalize)
        responses = frame.response.map(normalize)
        prefixes = frame.response.map(response_prefix)
        reasons = defaultdict(list)

        def mark(mask, reason):
            for sample_id in frame.loc[mask, "sample_id"]:
                reasons[int(sample_id)].append(reason)

        mark(frame.sample_id.isin(IDENTITY_ROWS.get(split, ())), "reviewed_identity_disclosure")
        mark(responses.isin(earlier_responses), "normalized_response_in_earlier_split")
        mark(prefixes.isin(earlier_prefixes), "response_token_prefix_in_earlier_split")
        repeated_prompts = frame.loc[prompts.isin(earlier_prompts), "prompt_group_id"]
        mark(frame.prompt_group_id.isin(repeated_prompts), "normalized_prompt_in_earlier_split")
        similar_prompts = frame.loc[
            frame.sample_id.isin(SIMILAR_PROMPT_ROWS.get(split, ())), "prompt_group_id"
        ]
        mark(frame.prompt_group_id.isin(similar_prompts), "reviewed_similar_prompt_in_earlier_split")

        retained = ~frame.sample_id.isin(reasons)
        cleaned[split] = frame.loc[retained]
        earlier_prompts.update(prompts[retained])
        earlier_responses.update(responses[retained])
        earlier_prefixes.update(prefixes[retained])
        exclusions.extend({"split": split, "sample_id": sample_id, "reasons": reasons[sample_id]}
                          for sample_id in sorted(reasons))

    OUTPUT.mkdir(exist_ok=False)
    output_hashes = {}
    for split, frame in cleaned.items():
        path = OUTPUT / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        output_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "source_sha256": SOURCE_HASHES,
        "split_sha256": output_hashes,
        "original_counts": {split: len(frame) for split, frame in frames.items()},
        "retained_counts": {split: len(frame) for split, frame in cleaned.items()},
        "excluded_rows": exclusions,
    }
    (OUTPUT / "cleaning_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest["retained_counts"])


if __name__ == "__main__":
    main()
