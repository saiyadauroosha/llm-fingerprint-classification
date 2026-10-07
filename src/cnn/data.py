"""Read the existing splits; never create new splits or learn a test vocabulary."""

import re
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD, UNK, SEP = '<PAD>', '<UNK>', '<SEP>'
TOKEN_PATTERN = re.compile(r'\n|\w+|[^\w\s]', re.UNICODE)


def tokenize(text):
    # Keep case, punctuation, Markdown symbols and newlines as style signals.
    # Runs of spaces/tabs are not represented by this baseline tokenizer.
    return TOKEN_PATTERN.findall(text.replace('\r\n', '\n').replace('\r', '\n'))


def input_tokens(row, mode, response_length, prompt_length):
    response = tokenize(row['response'])[:response_length]
    if mode == 'response':
        return response
    if mode == 'prompt_response':
        # Reserve the SAME response budget in both RQ1 and RQ2.
        return tokenize(row['prompt'])[:prompt_length] + [SEP] + response
    raise ValueError(f'Unknown input mode: {mode}')


def load_splits(data_dir):
    splits = {}
    required = {'sample_id', 'prompt_group_id', 'family', 'prompt', 'response'}
    for name in ('train', 'validation', 'test'):
        frame = pd.read_parquet(Path(data_dir) / f'{name}.parquet').reset_index(drop=True)
        if required - set(frame.columns):
            raise ValueError(f'{name}: missing columns {required - set(frame.columns)}')
        if frame.empty or frame[list(required)].isnull().any().any():
            raise ValueError(f'{name}: empty dataset or null required values.')
        for column in ('family', 'prompt', 'response'):
            if not frame[column].map(lambda s: isinstance(s, str) and bool(s.strip())).all():
                raise ValueError(f'{name}: {column} must contain non-empty strings.')
        if frame['sample_id'].duplicated().any():
            raise ValueError(f'{name}: duplicate sample_id values.')
        if 'split' in frame and not frame['split'].eq(name).all():
            raise ValueError(f'{name}: split column disagrees with the file name.')
        splits[name] = frame
    labels = set(splits['train']['family'])
    if len(labels) < 3:
        raise ValueError('This project requires at least three LLM families.')
    for name, frame in splits.items():
        if set(frame['family']) != labels:
            raise ValueError(f'{name}: all training families must be represented.')
    for first, second in combinations(splits, 2):
        for key in ('sample_id', 'prompt_group_id'):
            if set(splits[first][key]) & set(splits[second][key]):
                raise ValueError(f'{key} overlaps between {first} and {second}.')
        normalize = lambda value: ' '.join(value.split()).casefold()
        if set(splits[first]['prompt'].map(normalize)) & set(splits[second]['prompt'].map(normalize)):
            warnings.warn(f'Normalized prompt text overlaps: {first}/{second}. Review the prepared splits.')
        overlap = set(splits[first]['response']) & set(splits[second]['response'])
        if overlap:
            warnings.warn(f'{len(overlap)} identical response texts cross {first}/{second}; splits preserved.')
    return splits


def build_vocab(token_lists, max_vocab=30000, min_frequency=2):
    counts = Counter(token for tokens in token_lists for token in tokens)
    vocab = {PAD: 0, UNK: 1, SEP: 2}
    # Lexical tie-break keeps vocabulary ordering reproducible.
    for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(vocab) >= max_vocab:
            break
        if count >= min_frequency and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


class TextDataset(Dataset):
    def __init__(self, token_lists, families, vocab, label_to_id):
        self.ids = [torch.tensor([vocab.get(t, 1) for t in tokens] or [1], dtype=torch.long)
                    for tokens in token_lists]
        self.labels = [label_to_id[label] for label in families]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.ids[index], self.labels[index]


def collate_batch(batch):
    ids, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    return pad_sequence(ids, batch_first=True, padding_value=0), lengths, torch.tensor(labels)
