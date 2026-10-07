"""Kim-style TextCNN: embedding -> parallel Conv1d -> max pool -> classifier."""

import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=100,
                 num_filters=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in self.kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def forward(self, token_ids, lengths):
        # token_ids: [batch, sequence]; embedding: [batch, channels, sequence]
        if token_ids.size(1) < max(self.kernel_sizes):
            token_ids = F.pad(token_ids, (0, max(self.kernel_sizes) - token_ids.size(1)))
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for kernel, conv in zip(self.kernel_sizes, self.convs):
            features = F.relu(conv(embedded))
            # Exclude padded windows so predictions do not depend on batch mates.
            # For a text shorter than a kernel, allow one zero-padded window.
            valid_windows = (lengths - kernel + 1).clamp(min=1)
            positions = torch.arange(features.size(2), device=features.device)
            mask = positions[None, :] >= valid_windows[:, None]
            features = features.masked_fill(mask[:, None, :], float('-inf'))
            pooled.append(features.amax(dim=2))
        features = self.dropout(torch.cat(pooled, dim=1))
        # CrossEntropyLoss takes logits; apply softmax only for prediction.
        return self.classifier(features)


def initialize_glove(model, vocab, path):
    """Load a local GloVe text file; keep unmatched words randomly initialized."""
    dimension = model.embedding.embedding_dim
    matched = 0
    with open(path, encoding='utf-8') as handle, torch.no_grad():
        for line in handle:
            parts = line.rstrip().split()
            if not parts or parts[0] not in vocab or vocab[parts[0]] < 3:
                continue
            if len(parts) != dimension + 1:
                raise ValueError(f'GloVe dimension must match embedding_dim={dimension}.')
            vector = torch.tensor([float(x) for x in parts[1:]])
            if not torch.isfinite(vector).all():
                raise ValueError('GloVe file contains a non-finite vector.')
            model.embedding.weight[vocab[parts[0]]].copy_(vector)
            matched += 1
        model.embedding.weight[0].zero_()
    if not matched:
        raise ValueError('No GloVe words matched the training vocabulary.')
    return matched
