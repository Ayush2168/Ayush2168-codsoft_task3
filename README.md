# CodSoft AI Internship – Task 3: Image Captioning

Combines **Computer Vision + NLP**: a pre-trained **ResNet50** (CNN) extracts image
features and an **LSTM (RNN)** decoder generates the caption word by word.

```
Image -> ResNet50 (frozen, ImageNet) -> Linear+BN -> LSTM decoder -> "a dog runs through the grass"
```

## Files
| File | Purpose |
|------|---------|
| `model.py` | `EncoderCNN` (ResNet50) + `DecoderRNN` (LSTM, greedy + beam search) |
| `data.py` | Vocabulary, Flickr8k dataset, image transforms |
| `train.py` | Training loop with validation, saves best checkpoint |
| `caption.py` | Caption any image with your trained model |
| `pretrained_demo.py` | Instant captions with a ready-made ViT+GPT-2 model (no training) |

## Setup
```bash
pip install -r requirements.txt
```
Download **Flickr8k** (Kaggle: "Flickr8k" – images + `captions.txt`) and arrange:
```
data/Images/*.jpg
data/captions.txt
```

## Option A – train your own (CNN + RNN)
```bash
python train.py --data_dir data --epochs 10      # GPU recommended (Colab/Kaggle is fine)
python caption.py my_photo.jpg --ckpt checkpoints/best.pt
```

## Option B – pre-trained model, no training
```bash
python pretrained_demo.py my_photo.jpg
```

## Notes
- The CNN is frozen; only the projection layer and LSTM are trained, so it trains fast.
- Beam search (default width 3) gives better captions than greedy decoding.
- Improvements to try: attention over spatial features, transformer decoder, BLEU score evaluation.

#codsoft #AI #internship
