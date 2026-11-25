"""Show how many takes each label has, and warn about common problems.

    python check_data.py

Run it before training. It reads only file names, so it is instant.
Take files are named  LABEL__person__timestamp.npz
"""
import argparse
from collections import Counter
from pathlib import Path

from utils import ALL_LABELS, DYNAMIC_LABELS, IDLE

STATIC_TARGET = 10    # same targets as collect_data.py
DYNAMIC_TARGET = 20
IDLE_TARGET = 20
TEST_TARGET = 3


def scan(folder: Path):
    """Return ({label: count}, {label: set(persons)}, set(file stems))."""
    counts, persons, stems = Counter(), {}, set()
    for f in folder.glob("*.npz"):
        parts = f.stem.split("__")
        label = parts[0]
        counts[label] += 1
        persons.setdefault(label, set()).add(parts[1] if len(parts) > 1 else "?")
        stems.add(f.stem)
    return counts, persons, stems


def target_for(label: str) -> int:
    if label == IDLE:
        return IDLE_TARGET
    return DYNAMIC_TARGET if label in DYNAMIC_LABELS else STATIC_TARGET


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/takes")
    p.add_argument("--test-dir", default="data/test")
    args = p.parse_args()

    train, persons, train_stems = scan(Path(args.data))
    test, _, test_stems = scan(Path(args.test_dir))

    print(f"{'label':10s} {'train':>5s} {'goal':>5s} {'test':>5s}  people / notes")
    need_more, need_test = [], []
    for label in ALL_LABELS:
        n, goal, nt = train.get(label, 0), target_for(label), test.get(label, 0)
        notes = ",".join(sorted(persons.get(label, [])))
        if n < goal:
            notes += "   <-- record more"
            need_more.append(label)
        if nt < TEST_TARGET:
            need_test.append(label)
        print(f"{label:10s} {n:5d} {goal:5d} {nt:5d}  {notes}")

    unknown = sorted(set(train) | set(test) - set(ALL_LABELS))
    unknown = [u for u in unknown if u not in ALL_LABELS]
    if unknown:
        print(f"\nNote: files exist for labels not in utils.py: {unknown}")

    overlap = train_stems & test_stems
    print(f"\nTotal: {sum(train.values())} training takes, {sum(test.values())} test takes")
    if overlap:
        print(f"WARNING: {len(overlap)} take(s) are in BOTH {args.data} and {args.test_dir}.")
        print("  The test score would be inflated. Move or delete the test folder, then record a fresh one.")
    if need_more:
        print(f"Need more training takes: {', '.join(need_more)}")
    if need_test:
        print(f"Need at least {TEST_TARGET} test takes: {', '.join(need_test)}")
    if not overlap and not need_more and not need_test:
        print("Everything looks ready - run: python train_model.py")


if __name__ == "__main__":
    main()
