"""Encoder (pre-trained ResNet50 CNN) + Decoder (LSTM RNN) image captioning model."""
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Pre-trained ResNet50 -> fixed-size image feature vector."""

    def __init__(self, embed_size, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop final fc
        for p in self.backbone.parameters():   # freeze: we only use it as feature extractor
            p.requires_grad = False
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(images).flatten(1)
        return self.bn(self.fc(feats))


class DecoderRNN(nn.Module):
    """LSTM that generates a caption word-by-word, conditioned on image features."""

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """features: (B,E)  captions: (B,T) incl. <start>... Returns logits (B,T,V).
        The image feature is fed as the first 'word'; the input is captions[:, :-1]."""
        emb = self.drop(self.embed(captions[:, :-1]))
        inputs = torch.cat([features.unsqueeze(1), emb], dim=1)  # (B,T,E)
        out, _ = self.lstm(inputs)
        return self.fc(self.drop(out))

    @torch.no_grad()
    def generate(self, features, start_id, end_id, max_len=20):
        """Greedy decoding for a batch of image features."""
        ids, states = [], None
        inputs = features.unsqueeze(1)
        for _ in range(max_len):
            out, states = self.lstm(inputs, states)
            nxt = self.fc(out.squeeze(1)).argmax(-1)
            ids.append(nxt)
            inputs = self.embed(nxt).unsqueeze(1)
        ids = torch.stack(ids, 1)
        return ids

    @torch.no_grad()
    def beam_search(self, feature, start_id, end_id, beam=3, max_len=20):
        """Beam search for a single image feature (shape (1,E)). Returns list of ids."""
        out, states = self.lstm(feature.unsqueeze(1))          # feed image
        logp = torch.log_softmax(self.fc(out.squeeze(1)), -1)
        # first real step starts from <start>
        inp = self.embed(torch.tensor([start_id], device=feature.device)).unsqueeze(1)
        out, states = self.lstm(inp, states)
        logp = torch.log_softmax(self.fc(out.squeeze(1)), -1)[0]
        top = logp.topk(beam)
        beams = [([i.item()], s.item(), states) for s, i in zip(top.values, top.indices)]
        done = []
        for _ in range(max_len - 1):
            cand = []
            for seq, score, st in beams:
                if seq[-1] == end_id:
                    done.append((seq, score)); continue
                inp = self.embed(torch.tensor([[seq[-1]]], device=feature.device))
                o, new_st = self.lstm(inp, st)
                lp = torch.log_softmax(self.fc(o.squeeze(1)), -1)[0]
                t = lp.topk(beam)
                for s, i in zip(t.values, t.indices):
                    cand.append((seq + [i.item()], score + s.item(), new_st))
            if not cand:
                break
            beams = sorted(cand, key=lambda x: x[1], reverse=True)[:beam]
        done += [(s, sc) for s, sc, _ in beams]
        best = max(done, key=lambda x: x[1] / len(x[0]))
        return best[0]
