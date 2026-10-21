"""Zero-training demo using a ready-made pre-trained captioning model
(ViT encoder + GPT-2 decoder from Hugging Face). Needs internet on first run.

    python pretrained_demo.py path/to/image.jpg
"""
import sys

import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

NAME = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(NAME).eval()
processor = ViTImageProcessor.from_pretrained(NAME)
tok = AutoTokenizer.from_pretrained(NAME)


def caption(path):
    pixels = processor(Image.open(path).convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        out = model.generate(pixels, max_length=20, num_beams=4)
    return tok.decode(out[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: {caption(p)}")
