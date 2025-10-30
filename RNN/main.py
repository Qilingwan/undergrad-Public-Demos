# main.py: RNN for Name Generation
# Author: Chen
# Description: This script implements an RNN model to generate names based on a dataset of existing names. It includes data loading, preprocessing, model definition, training, and functions for sampling new names with optional prefixes and constraints.
# Tip: This code was originally run on Kaggle with GPU acceleration. If running in a different environment, please adjust input/output paths (e.g., /kaggle/input/) and ensure CUDA availability.

import os, random, math, itertools, string, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

DATA_PATH = '/kaggle/input/namelist/names.txt'
EPOCHS        = 15      
BATCH_SIZE    = 256
EMB_SIZE      = 32
HIDDEN_SIZE   = 128
LR            = 3e-3
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED          = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

with open(DATA_PATH, encoding='utf-8') as f:
    rawNames = [line.strip() for line in f if line.strip()]
print(f'Total names read: {len(rawNames):,}')

def cleanName(n):
    return n.title()  # e.g. 'mArY-ann' -> 'Mary-Ann'

rawNames = [cleanName(n) for n in rawNames]
alphabet = sorted({ch for name in rawNames for ch in name})
SOS_TOKEN, EOS_TOKEN = '^', '$'
alphabet = [SOS_TOKEN, EOS_TOKEN] + alphabet
char2idx = {c:i for i,c in enumerate(alphabet)}
idx2char = {i:c for c,i in char2idx.items()}
VOCAB_SIZE = len(alphabet)
print('Alphabet (with SOS/EOS):', ''.join(alphabet))

class NameDataset(Dataset):
    def __init__(self, names):
        self.data = names

    def __len__(self):
        return len(self.data)

    def encode(self, name):
        return [char2idx[SOS_TOKEN]] + [char2idx[c] for c in name] + [char2idx[EOS_TOKEN]]

    def __getitem__(self, idx):
        idxSeq = self.encode(self.data[idx])
        x = torch.tensor(idxSeq[:-1], dtype=torch.long)  # input chars
        y = torch.tensor(idxSeq[1:], dtype=torch.long)   # targets (next char)
        return x, y

def collate(batch):
    xs, ys = zip(*batch)
    lens = [len(x) for x in xs]
    maxlen = max(lens)
    padValue = char2idx[EOS_TOKEN]  # pad with EOS
    paddedX = torch.full((len(xs), maxlen), padValue, dtype=torch.long)
    paddedY = torch.full((len(xs), maxlen), padValue, dtype=torch.long)
    for i,(x,y,l) in enumerate(zip(xs, ys, lens)):
        paddedX[i, :l] = x
        paddedY[i, :l] = y
    return paddedX, paddedY, torch.tensor(lens)

trainLoader = DataLoader(NameDataset(rawNames), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate, drop_last=True)

class CharRNN(nn.Module):
    def __init__(self, vocabSize, embSize, hiddenSize):
        super().__init__()
        self.embedding = nn.Embedding(vocabSize, embSize)
        self.gru = nn.GRU(embSize, hiddenSize, batch_first=True)
        self.fc = nn.Linear(hiddenSize, vocabSize)

    def forward(self, x, h=None):
        emb = self.embedding(x)
        out, h = self.gru(emb, h)
        logits = self.fc(out)           # (B, T, V)
        return logits, h

model = CharRNN(VOCAB_SIZE, EMB_SIZE, HIDDEN_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=char2idx[EOS_TOKEN])
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def trainEpoch(epoch):
    model.train()
    totalLoss, n = 0,0
    for bx, by, lens in trainLoader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(bx)
        loss = criterion(logits.view(-1, VOCAB_SIZE), by.view(-1))
        loss.backward()
        optimizer.step()
        totalLoss += loss.item() * bx.size(0)
        n += bx.size(0)
    print(f'Epoch {epoch}: loss = {totalLoss/n:.4f}')

for ep in range(1, EPOCHS+1):
    trainEpoch(ep)

@torch.no_grad()
def sampleName(prefix='', maxLen=20, temp=0.8,
                returnLogprobs=False):
    model.eval()
    prefix = cleanName(prefix)
    inputIdxs = [char2idx[SOS_TOKEN]] + [char2idx[c] for c in prefix]
    inputTensor = torch.tensor([inputIdxs], dtype=torch.long, device=DEVICE)
    hidden = None
    outputs, probsLog = [], []
    # Feed known prefix
    logits, hidden = model(inputTensor)
    # Generate subsequent characters
    chIdx = None
    for t in range(maxLen):
        lastLogits = logits[0, -1] / temp
        prob = torch.softmax(lastLogits, dim=-1)
        chIdx = torch.multinomial(prob, 1).item()
        outputs.append(idx2char[chIdx])
        probsLog.append(prob[chIdx].log().item())
        if chIdx == char2idx[EOS_TOKEN]: break
        inputTensor = torch.tensor([[chIdx]], dtype=torch.long, device=DEVICE)
        logits, hidden = model(inputTensor, hidden)
    name = prefix + ''.join(outputs)
    name = name.split('$')[0]  # truncate at EOS if present
    if returnLogprobs:
        return name, probsLog, prefix
    return name

def visualizeGen(name, probsLog, prefix=''):
    fig, axs = plt.subplots(1, len(name)+1, figsize=(14,4))
    fig.suptitle(f'Generated name: {name}')
    prevLogp = 0.
    preLen = len(prefix)
    for i, (ch, logp) in enumerate(zip(name, probsLog)):
        bar = axs[i].bar(['Prev', 'This'], [prevLogp, logp])
        axs[i].bar_label(bar, fmt='%.2f')
        axs[i].set_title(ch if i >= preLen else f'[{ch}]')
        prevLogp += logp
    axs[-1].bar('Total', prevLogp)
    axs[-1].bar_label(axs[-1].containers[0], fmt='%.2f')
    axs[-1].set_title('LogProb')
    plt.tight_layout()

def genWithConstraints(prefix='', suffix='', temp=0.8, nTries=500):
    cands = set()
    for _ in range(nTries):
        name = sampleName(prefix, temp=temp)
        if name.endswith(suffix):
            cands.add(name)
    return sorted(cands)

if __name__ == '__main__':
    demoPrefix = 'El'
    name, probs, pre = sampleName(demoPrefix, returnLogprobs=True)
    print('Generated:', name)
    visualizeGen(name, probs, pre)

    cands = genWithConstraints(prefix='Mc', suffix='son', temp=0.9)
    for i,nm in enumerate(cands[:10],1):
        print(f'{i:2d}. {nm}')

    fav = cands[0] if cands else sampleName('Kobe')# What can I say
    print(f'\n[Favorite] {fav}')
    favName, favProbs, favPre = sampleName(fav[:2], returnLogprobs=True)
    visualizeGen(favName, favProbs, favPre)