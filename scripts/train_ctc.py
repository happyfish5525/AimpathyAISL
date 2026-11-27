from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

"""Train the CTC fingerspelling model (bonus 1).

    python train_ctc.py                     # data/takes for training, data/test for the test
    python train_ctc.py --epochs 5          # quick smoke test

What happens
------------
* Training sequences are stitched from isolated takes (see seq_data.py) and are
  RE-GENERATED EVERY EPOCH, so the model never sees the same sequence twice.
* Validation uses sequences stitched from held-out takes (same signer).
* TEST uses sequences stitched from data/test, i.e. a signer the model never
  trained on. Stitched test sequences are still easier than real signing,
  so this is NOT the final number: the final number comes from real recorded
  words (next step). It is useful for comparing runs and catching bugs.
* Score = character error rate (CER, lower is better) and exact-word accuracy.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from ctc_model import SeqNet, load_checkpoint, predict_paths, save_checkpoint
from pathlib import Path

from seq_data import (FPS, T, VOCAB, WINDOW_SEC, cer, collapse_path, exclude_words,
                      load_takes, load_word_takes, make_bank, make_real_set, make_set,
                      split_takes, tokens_to_text)
from utils import FEAT_DIM

HIDDEN = 128


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/takes")
    p.add_argument("--test-dir", default="data/test")
    p.add_argument("--out", default="asl_ctc.pth")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--steps", type=int, default=50, help="batches per epoch")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument("--exclude-words-dir", default="data/words_test",
                   help="real test words recorded here are removed from the stitched training words")
    p.add_argument("--words-dir", default="data/words_train",
                   help="real words from a TRAINING signer (collect_words.py); optional")
    p.add_argument("--real-frac", type=float, default=0.25,
                   help="share of each epoch's examples that are real words (when --words-dir exists)")
    p.add_argument("--real-val-every", type=int, default=6,
                   help="hold out every Nth training word for validation (0 = train on all of them)")
    p.add_argument("--init", default=None,
                   help="continue from this checkpoint (e.g. asl_ctc.pth) instead of starting fresh")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def decode(paths):
    """Per-frame argmax paths -> list of strings (greedy CTC decoding)."""
    return [tokens_to_text([tok for _, tok in collapse_path(p)]) for p in paths]


def score(model, x, targets):
    refs = [tokens_to_text(t) for t in targets]
    hyps = decode(predict_paths(model, x))
    exact = float(np.mean([r == h for r, h in zip(refs, hyps)]))
    return cer(refs, hyps), exact, refs, hyps


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    wdir = Path(args.exclude_words_dir)
    if wdir.exists():
        test_words = {f.stem.split("__")[0] for f in wdir.glob("*.npz")}
        print(f"excluded {exclude_words(test_words)} real test words from the stitched training words")
    takes = load_takes(args.data)
    word_train, word_val = [], []
    if Path(args.words_dir).exists():
        words = load_word_takes(args.words_dir)
        if wdir.exists():                    # never train on a word that is in the test set
            clash = sorted({w["label"] for w in words} & test_words)
            if clash:
                print(f"WARNING: dropping training words that are also test words: {clash}")
            words = [w for w in words if w["label"] not in test_words]
        labels = sorted({w["label"] for w in words})
        val_labels = set(labels[::args.real_val_every]) if args.real_val_every > 0 else set()        # whole words held out, so validation is on unseen words
        word_val = [w for w in words if w["label"] in val_labels]
        word_train = [w for w in words if w["label"] not in val_labels]
        print(f"real words: {len(word_train)} train takes / {len(word_val)} validation takes "
              f"(validation words: {sorted(val_labels)})")
    if not takes:
        raise SystemExit(f"No usable takes in '{args.data}'.")
    train_takes, val_takes = split_takes(takes, rng)
    train_bank, val_bank = make_bank(train_takes), make_bank(val_takes)
    missing = [t for t in VOCAB[1:] if not train_bank.get(t)]
    if missing:
        print(f"warning: no training takes for {missing}; they can never be output")
    print(f"{len(takes)} takes -> {len(train_takes)} train / {len(val_takes)} validation")

    x_val, y_val = make_set(val_bank, args.val_size, seed=args.seed + 1, augment=False)
    x_rval, y_rval = make_real_set(word_val, seed=args.seed + 2) if word_val else (None, [])
    if y_rval:
        print(f"{len(y_rval)} real validation words usable (model selection uses these)")

    model = SeqNet(FEAT_DIM, len(VOCAB), HIDDEN)
    if args.init:
        model.load_state_dict(load_checkpoint(args.init)[0].state_dict())
        print(f"continuing from {args.init}")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    best_cer, best_state, stale = 9.9, None, 0
    n_per_epoch = args.steps * args.batch_size
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        x, ys = make_set(train_bank, n_per_epoch, seed=int(rng.integers(1 << 30)),
                         word_takes=word_train, real_frac=args.real_frac if word_train else 0.0)
        x = torch.from_numpy(x)
        model.train()
        total = 0.0
        for i in range(0, n_per_epoch, args.batch_size):
            xb, yb = x[i:i + args.batch_size], ys[i:i + args.batch_size]
            targets = torch.tensor([t for y in yb for t in y], dtype=torch.long)
            tgt_len = torch.tensor([len(y) for y in yb], dtype=torch.long)
            in_len = torch.full((len(yb),), T, dtype=torch.long)
            log_probs = model(xb).log_softmax(dim=-1).transpose(0, 1)   # (T, B, C)
            loss = ctc(log_probs, targets, in_len, tgt_len)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        train_loss = total / (n_per_epoch // args.batch_size)

        val_cer, val_exact, _, _ = score(model, x_val, y_val)
        real_note = ""
        if y_rval:
            real_cer, real_exact, _, _ = score(model, x_rval, y_rval)
            real_note = f"  real-word CER {real_cer * 100:5.1f}%"
            # only ~20 real validation takes: average with the stitched score so one
            # noisy number cannot pick the model or stop training too early
            val_cer = 0.5 * (val_cer + real_cer)
            val_exact = 0.5 * (val_exact + real_exact)
        sched.step(train_loss)   # not val CER: it sits at 100% until the model starts emitting letters
        print(f"epoch {epoch:3d}  ctc loss {train_loss:6.3f}  val CER {val_cer * 100:5.1f}%  "
              f"val exact {val_exact * 100:5.1f}%{real_note}  ({time.time() - t0:.1f}s)")
        if val_cer < best_cer:
            best_cer, stale = val_cer, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif best_cer < 1.0:     # only count 'no improvement' once letters are being produced
            stale += 1
            if stale >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    config = {"feat_dim": FEAT_DIM, "hidden": HIDDEN, "fps": FPS, "window_sec": WINDOW_SEC}
    save_checkpoint(args.out, model, config)
    print(f"\nSaved best model to {args.out}  (val CER {best_cer * 100:.1f}%)")

    test_takes = load_takes(args.test_dir)
    if test_takes:
        x_t, y_t = make_set(make_bank(test_takes), args.val_size, seed=999, augment=False)
        t_cer, t_exact, refs, hyps = score(model, x_t, y_t)
        people = sorted({t["person"] for t in test_takes})
        print(f"\nTEST on stitched sequences from unseen signer(s) {people}:")
        print(f"  CER {t_cer * 100:.1f}%   exact-word accuracy {t_exact * 100:.1f}%")
        print("  examples (truth -> model):")
        for r, h in list(zip(refs, hyps))[:10]:
            print(f"    {r!r:14} -> {h!r}" + ("" if r == h else "   <-- wrong"))
    print("\nReminder: stitched test sequences are easier than real signing. "
          "Record real words (collect_words.py) for the final comparison.")


if __name__ == "__main__":
    main()