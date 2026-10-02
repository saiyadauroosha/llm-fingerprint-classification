#Load the fixed splits
import pandas as pd
train = pd.read_parquet("data/splits/train.parquet")
val   = pd.read_parquet("data/splits/validation.parquet")
test  = pd.read_parquet("data/splits/test.parquet")

label2id = {"GPT": 0, "Claude": 1, "Gemini": 2}  # check the exact strings in `family`

#Tokenize and build the vocabulary from train only
import re
from collections import Counter

def tokenize(text):
    # words, newlines, and individual punctuation/markdown symbols as tokens
    return re.findall(r"\w+|\n|[^\w\s]", text)

counts = Counter(tok for t in train["response"] for tok in tokenize(t))
vocab = {"<pad>": 0, "<unk>": 1}
for tok, c in counts.items():
    if c >= 2:
        vocab[tok] = len(vocab)

#Encode, truncate/pad, and wrap in a Dataset
import torch
from torch.utils.data import Dataset, DataLoader

MAX_LEN = 512

def encode(text):
    ids = [vocab.get(t, 1) for t in tokenize(text)][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

class TextDS(Dataset):
    def __init__(self, df, col="response"):
        self.x = [encode(t) for t in df[col]]
        self.y = [label2id[f] for f in df["family"]]
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])

train_dl = DataLoader(TextDS(train), batch_size=64, shuffle=True)
val_dl   = DataLoader(TextDS(val), batch_size=64)

#The model
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, n_filters=128,
                 kernels=(3, 4, 5), n_classes=3, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, n_filters, k) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernels), n_classes)

    def forward(self, x):                    # x: (batch, seq_len)
        e = self.emb(x).transpose(1, 2)      # (batch, emb_dim, seq_len)
        pooled = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1)))

#Train, select on validation, test once
model = TextCNN(len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for x, y in train_dl:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    # then: model.eval(), compute validation loss/accuracy under torch.no_grad(),
    # save the checkpoint with the best validation score