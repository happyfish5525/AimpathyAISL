"""Sequence model: reads a stream of hand frames, writes letters in order.

Difference from model.py (SignNet)
----------------------------------
SignNet squeezes a whole window into ONE label. SeqNet keeps one output per
time step (25 per second). Trained with CTC loss, it learns by itself WHEN a
letter happens, that a letter held for many frames is still one letter, and
that a repeated letter (LL) needs a short gap. Those jobs were previously
done by hand-written thresholds in recognizer.py.

Output classes: index 0 = blank ("no new letter here"), then A-Z, then SPACE.
"""
import numpy as np
import torch
import torch.nn as nn

from seq_data import VOCAB


class SeqNet(nn.Module):
    """BatchNorm -> 2x 1D-conv -> 2-layer bidirectional GRU -> per-frame logits.

    Input : (batch, T, feat_dim)      Output: (batch, T, num_classes) logits
    The convolutions (padding keeps T unchanged) give each frame a short
    local context; the BiGRU gives it the whole window as context.
    """

    def __init__(self, feat_dim: int, num_classes: int = len(VOCAB),
                 hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.BatchNorm1d(feat_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.norm(x.transpose(1, 2))          # (B, F, T)
        x = self.drop(self.conv(x).transpose(1, 2))   # (B, T, H)
        out, _ = self.gru(x)                      # (B, T, 2H)
        return self.head(self.drop(out))          # (B, T, C)


def save_checkpoint(path, model, config):
    torch.save({"model_state": model.state_dict(), "vocab": list(VOCAB),
                "config": config}, path)


def load_checkpoint(path):
    """Return (model in eval mode, config dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    if list(ckpt["vocab"]) != list(VOCAB):
        raise ValueError("Checkpoint vocabulary differs from seq_data.VOCAB")
    model = SeqNet(cfg["feat_dim"], len(VOCAB), cfg["hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_paths(model, feats: np.ndarray) -> np.ndarray:
    """feats (N, T, F) -> per-frame argmax paths (N, T) of token indices."""
    model.eval()
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    return model(x).argmax(dim=-1).numpy()
