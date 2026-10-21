"""Train the CNN-RNN captioning model on Flickr8k.

    python train.py --data_dir data --epochs 10
"""
import argparse
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import FlickrDataset, Vocabulary, collate, eval_tf, read_captions, train_tf
from model import DecoderRNN, EncoderCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = read_captions(os.path.join(a.data_dir, "captions.txt"))
    images = sorted({p[0] for p in pairs})
    random.seed(0); random.shuffle(images)
    val_imgs = set(images[: int(0.1 * len(images))])        # 10% images for validation
    train_pairs = [p for p in pairs if p[0] not in val_imgs]
    val_pairs = [p for p in pairs if p[0] in val_imgs]

    vocab = Vocabulary(min_freq=3)
    vocab.build([c for _, c in train_pairs])
    print(f"vocab={len(vocab)}  train={len(train_pairs)}  val={len(val_pairs)}  device={dev}")

    img_dir = os.path.join(a.data_dir, "Images")
    tl = DataLoader(FlickrDataset(img_dir, train_pairs, vocab, train_tf), a.batch_size,
                    shuffle=True, collate_fn=collate, num_workers=a.workers)
    vl = DataLoader(FlickrDataset(img_dir, val_pairs, vocab, eval_tf), a.batch_size,
                    collate_fn=collate, num_workers=a.workers)

    enc, dec = EncoderCNN(a.embed).to(dev), DecoderRNN(a.embed, a.hidden, len(vocab)).to(dev)
    params = list(enc.fc.parameters()) + list(enc.bn.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    os.makedirs(a.out, exist_ok=True)
    best = float("inf")

    for ep in range(1, a.epochs + 1):
        enc.train(); dec.train()
        tot = 0
        for i, (imgs, caps, _) in enumerate(tl):
            imgs, caps = imgs.to(dev), caps.to(dev)
            logits = dec(enc(imgs), caps)                    # (B,T,V)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), caps.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            tot += loss.item()
            if i % 50 == 0:
                print(f"epoch {ep} step {i}/{len(tl)} loss {loss.item():.3f}")

        enc.eval(); dec.eval(); vtot = 0
        with torch.no_grad():
            for imgs, caps, _ in vl:
                imgs, caps = imgs.to(dev), caps.to(dev)
                logits = dec(enc(imgs), caps)
                vtot += loss_fn(logits.reshape(-1, logits.size(-1)), caps.reshape(-1)).item()
        vloss = vtot / len(vl)
        print(f"=== epoch {ep}: train {tot/len(tl):.3f}  val {vloss:.3f}")
        if vloss < best:
            best = vloss
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                        "itos": vocab.itos, "embed": a.embed, "hidden": a.hidden},
                       os.path.join(a.out, "best.pt"))
            print("saved best checkpoint")


if __name__ == "__main__":
    main()
