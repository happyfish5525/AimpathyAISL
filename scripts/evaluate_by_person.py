from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

"""Score a trained model separately for each person in a test folder.

    python evaluate_by_person.py                      # asl_model.pth on data/test
    python evaluate_by_person.py --model asl_model.pth --test-dir data/test

Take files are named  LABEL__person__timestamp.npz  (e.g. HELLO__leyu_test__1795151000779.npz),
so the person is the middle part. Put this file next to train_model.py.

Accuracy uses the same rule as train_model.py: look-alike pairs
(0/O, 1/D, 2/V, 6/W, 9/F) count as one class because the ABC/123 mode decides between them.
"""
import argparse
from collections import defaultdict

import torch

from model import load_checkpoint
from train_model import load_takes, make_eval_windows, report
from utils import canonical


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="asl_model.pth")
    p.add_argument("--test-dir", default="data/test")
    args = p.parse_args()

    model, classes = load_checkpoint(args.model)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    canon_idx = torch.tensor([class_to_idx.get(canonical(c), i)
                              for i, c in enumerate(classes)])

    takes = [t for t in load_takes(args.test_dir) if t["label"] in class_to_idx]
    if not takes:
        raise SystemExit(f"No usable takes in '{args.test_dir}'.")

    by_person = defaultdict(list)
    for t in takes:
        by_person[t["id"].split("__")[1]].append(t)   # id = LABEL__person__timestamp

    for person, group in sorted(by_person.items()):
        x, y = make_eval_windows(group, class_to_idx)
        report(f"TEST - {person} ({len(group)} takes)", model, x, y, classes, canon_idx)


if __name__ == "__main__":
    main()
