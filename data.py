"""Vocabulary + Flickr8k dataset loader.

Expected layout (Kaggle "Flickr8k" dataset):
    data/Images/*.jpg
    data/captions.txt        # CSV with header: image,caption
"""
import csv
import re
from collections import Counter

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


class Vocabulary:
    def __init__(self, min_freq=3):
        self.min_freq = min_freq
        self.itos = [PAD, START, END, UNK]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def build(self, captions):
        counts = Counter(w for c in captions for w in tokenize(c))
        for w, n in counts.items():
            if n >= self.min_freq and w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)

    def encode(self, text):
        unk = self.stoi[UNK]
        return [self.stoi[START]] + [self.stoi.get(w, unk) for w in tokenize(text)] + [self.stoi[END]]

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.itos[i]
            if w == END:
                break
            if w not in (PAD, START):
                words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.itos)


# ResNet expects ImageNet-normalised 224x224 input
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def read_captions(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(r["image"], r["caption"]) for r in rows]


class FlickrDataset(Dataset):
    def __init__(self, img_dir, pairs, vocab, transform=train_tf):
        self.img_dir, self.pairs, self.vocab, self.tf = img_dir, pairs, vocab, transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name, cap = self.pairs[idx]
        img = self.tf(Image.open(f"{self.img_dir}/{name}").convert("RGB"))
        return img, torch.tensor(self.vocab.encode(cap))


def collate(batch):
    imgs, caps = zip(*batch)
    lengths = torch.tensor([len(c) for c in caps])
    return torch.stack(imgs), pad_sequence(caps, batch_first=True, padding_value=0), lengths
