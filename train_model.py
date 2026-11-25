"""Train the sign model from recorded takes.

    python train_model.py                      # uses data/takes
    python train_model.py --test-dir data/test # also score on a separate test set

What is different from the first version
----------------------------------------
* The validation split is made BY TAKE, so the score is not inflated by
  near-identical neighbouring windows.
* Windows are re-cut and re-augmented every epoch (rotation, scale, noise,
  mirroring, speed change, dropped frames).
* The best epoch (by validation accuracy) is saved, not the last one.
* Look-alike pairs (0/O, 2/V, 6/W, 9/F, 1/D) are scored as one class, because
  the mode toggle decides between them at recognition time.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import SignNet, save_checkpoint
from utils import (ALL_LABELS, FEAT_DIM, IDLE, SEQ_LEN, STATIC_WINDOW_SEC,
                   DYNAMIC_WINDOW_SEC, DYNAMIC_LABELS, augment_window,
                   build_features, canonical, sample_window)

HIDDEN = 96


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/takes")
    p.add_argument("--test-dir", default="data/test")
    p.add_argument("--out", default="asl_model.pth")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--windows-per-take", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_takes(folder):
    takes = []
    for f in sorted(Path(folder).glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        takes.append({"id": f.stem, "label": str(d["label"]),
                      "times": d["times"], "feats": d["feats"]})
    return takes


def split_by_take(takes, rng, val_frac=0.2):
    """Hold out whole takes per class (a class needs >= 3 takes to get a val take)."""
    by_label = {}
    for t in takes:
        by_label.setdefault(t["label"], []).append(t)
    train, val = [], []
    for label, group in by_label.items():
        order = rng.permutation(len(group))
        n_val = max(1, round(len(group) * val_frac)) if len(group) >= 3 else 0
        val += [group[i] for i in order[:n_val]]
        train += [group[i] for i in order[n_val:]]
        if n_val == 0:
            print(f"  warning: '{label}' has only {len(group)} takes - not validated")
    return train, val


def make_windows(takes, class_to_idx, rng, per_take, augment):
    xs, ys = [], []
    for t in takes:
        for _ in range(per_take):
            raw = sample_window(t["times"], t["feats"], t["label"], rng)
            if augment:
                raw = augment_window(raw, rng)
            xs.append(build_features(raw))
            ys.append(class_to_idx[t["label"]])
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys)


def make_eval_windows(takes, class_to_idx, per_take=4, seed=123):
    """Fixed, un-augmented windows at the live window lengths."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for t in takes:
        for k in range(per_take):
            if t["label"] == IDLE:
                dur = STATIC_WINDOW_SEC if k % 2 == 0 else DYNAMIC_WINDOW_SEC
            elif t["label"] in DYNAMIC_LABELS:
                dur = DYNAMIC_WINDOW_SEC
            else:
                dur = STATIC_WINDOW_SEC
            raw = sample_window(t["times"], t["feats"], t["label"], rng, duration=dur)
            xs.append(build_features(raw))
            ys.append(class_to_idx[t["label"]])
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys)


@torch.no_grad()
def evaluate(model, x, y, canon_idx, criterion=None):
    model.eval()
    logits = model(x)
    loss = criterion(logits, y).item() if criterion is not None else 0.0
    pred = logits.argmax(dim=1)
    ok = canon_idx[pred] == canon_idx[y]          # look-alikes count as correct
    return loss, ok.float().mean().item(), pred


def report(name, model, x, y, classes, canon_idx):
    _, acc, pred = evaluate(model, x, y, canon_idx)
    print(f"\n{name}: accuracy {acc * 100:.1f}%  ({len(y)} windows)")
    pred, truth, canon = pred.numpy(), y.numpy(), canon_idx.numpy()
    per_class = {}
    confusions = Counter()
    for p_i, t_i in zip(pred, truth):
        right = bool(canon[p_i] == canon[t_i])
        per_class.setdefault(classes[t_i], []).append(right)
        if not right:
            confusions[(classes[t_i], classes[p_i])] += 1
    worst = sorted(per_class.items(), key=lambda kv: np.mean(kv[1]))[:8]
    print("  weakest classes:", ", ".join(f"{c} {np.mean(v) * 100:.0f}%" for c, v in worst))
    if confusions:
        print("  most common mistakes (true -> predicted):")
        for (t, p), n in confusions.most_common(8):
            print(f"    {t:10s} -> {p:10s} x{n}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    takes = load_takes(args.data)
    if not takes:
        raise SystemExit(f"No takes found in '{args.data}'. Run collect_data.py first.")
    present = {t["label"] for t in takes}
    classes = [c for c in ALL_LABELS if c in present]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    if IDLE not in present:
        print("warning: no IDLE takes - the model will force a letter whenever a hand is visible")
    print(f"{len(takes)} takes, {len(classes)} classes")

    train_takes, val_takes = split_by_take(takes, rng)
    print(f"train takes: {len(train_takes)}   validation takes: {len(val_takes)}")

    canon_idx = torch.tensor([class_to_idx.get(canonical(c), i)
                              for i, c in enumerate(classes)])
    counts = Counter(t["label"] for t in train_takes)
    mean_count = np.mean(list(counts.values()))
    weights = torch.tensor([(mean_count / counts[c]) ** 0.5 for c in classes],
                           dtype=torch.float32)  # gentle boost for rare classes
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = SignNet(FEAT_DIM, len(classes), HIDDEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    x_val = y_val = None
    if val_takes:
        x_val, y_val = make_eval_windows(val_takes, class_to_idx)

    best_acc, best_loss, best_state, stale = -1.0, 1e9, None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        x, y = make_windows(train_takes, class_to_idx, rng, args.windows_per_take, True)
        model.train()
        perm = torch.randperm(len(y))
        total = 0.0
        for i in range(0, len(y), args.batch_size):
            idx = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        train_loss = total / len(y)

        if x_val is not None:
            val_loss, val_acc, _ = evaluate(model, x_val, y_val, canon_idx, criterion)
            scheduler.step(val_loss)
        else:
            val_loss, val_acc = train_loss, 0.0
        print(f"epoch {epoch:3d}  train loss {train_loss:.3f}  "
              f"val loss {val_loss:.3f}  val acc {val_acc * 100:5.1f}%  "
              f"({time.time() - t0:.1f}s)")

        improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)
        if improved or x_val is None:
            best_acc, best_loss, stale = val_acc, val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    config = {"feat_dim": FEAT_DIM, "seq_len": SEQ_LEN, "hidden": HIDDEN}
    save_checkpoint(args.out, model, classes, config)
    print(f"\nSaved best model to {args.out}")

    if x_val is not None:
        report("VALIDATION (held-out takes)", model, x_val, y_val, classes, canon_idx)
    test_takes = [t for t in load_takes(args.test_dir) if t["label"] in class_to_idx] \
        if Path(args.test_dir).exists() else []
    if test_takes:
        x_t, y_t = make_eval_windows(test_takes, class_to_idx)
        report("TEST (separate recording session)", model, x_t, y_t, classes, canon_idx)
    else:
        print("\n(no test set found - record one with: python collect_data.py --out data/test)")


if __name__ == "__main__":
    main()
