"""Generate a caption for an image with your trained model.

    python caption.py path/to/image.jpg --ckpt checkpoints/best.pt
"""
import argparse

import torch
from PIL import Image

from data import END, START, Vocabulary, eval_tf
from model import DecoderRNN, EncoderCNN


def load(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location=dev)
    vocab = Vocabulary(); vocab.itos = ck["itos"]; vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    enc = EncoderCNN(ck["embed"], pretrained=False).to(dev)
    dec = DecoderRNN(ck["embed"], ck["hidden"], len(vocab)).to(dev)
    enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
    return enc.eval(), dec.eval(), vocab


def caption_image(path, enc, dec, vocab, dev, beam=3):
    img = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        feat = enc(img)
        ids = dec.beam_search(feat, vocab.stoi[START], vocab.stoi[END], beam=beam)
    return vocab.decode(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--beam", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, dec, vocab = load(a.ckpt, dev)
    print("Caption:", caption_image(a.image, enc, dec, vocab, dev, a.beam))
