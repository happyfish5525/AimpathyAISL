"""Neural network used for both training and live recognition."""
import numpy as np
import torch
import torch.nn as nn


class SignNet(nn.Module):
    """1D-conv -> bidirectional GRU -> pooled classifier.

    Input : (batch, SEQ_LEN, FEAT_DIM) feature windows
    Output: (batch, num_classes) logits

    The GRU gives the model memory of how the hand moved over the window; the
    mean+max pooling over time means a sign does not have to line up exactly
    with the end of the window. It is small enough for a laptop CPU.
    """

    def __init__(self, feat_dim: int, num_classes: int,
                 hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.BatchNorm1d(feat_dim)  # shape, position and velocity have different scales
        self.conv = nn.Sequential(nn.Conv1d(feat_dim, hidden, 3, padding=1), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.norm(x.transpose(1, 2))      # (B, F, T)
        x = self.conv(x).transpose(1, 2)      # (B, T, H)
        out, _ = self.gru(x)                  # (B, T, 2H)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(pooled)


def save_checkpoint(path, model, classes, config):
    torch.save({"model_state": model.state_dict(),
                "classes": list(classes),
                "config": config}, path)


def load_checkpoint(path):
    """Return (model in eval mode, list of class names)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = SignNet(cfg["feat_dim"], len(ckpt["classes"]), cfg["hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]


@torch.no_grad()
def predict_probs(model, feats: np.ndarray) -> np.ndarray:
    """feats: (N, T, F) numpy -> (N, num_classes) probabilities."""
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    return torch.softmax(model(x), dim=1).numpy()
