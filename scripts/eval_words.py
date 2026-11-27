from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

"""Compare the two pipelines on REAL recorded words (the honest test).

    python eval_words.py                         # CTC model vs the old pipeline
    python eval_words.py --verbose               # also list every wrong take

Both pipelines see each recorded take exactly like the live app would: frames
arrive in time order and nothing from the future is visible.

  OLD : SignNet window classifier + recognizer.py rules  (mode ABC)
  CTC : SeqNet + CTC + stream_decoder.py

Scores per take: character error rate (CER, lower is better; counts missing,
extra and wrong letters) and exact-word accuracy (every letter right).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ctc_model import load_checkpoint as load_ctc
from ctc_model import predict_paths
from seq_data import T, WINDOW_SEC, WORDS, cer
from stream_decoder import DELAY_SEC, StreamingDecoder
from utils import DYNAMIC_LABELS, DYNAMIC_WINDOW_SEC, build_features, resample_window


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--words-dir", default="data/words_test")
    p.add_argument("--ctc", default="asl_ctc.pth")
    p.add_argument("--old", default="asl_model.pth", help="old SignNet checkpoint ('' to skip)")
    p.add_argument("--step", type=float, default=0.2, help="seconds between CTC runs")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_word_takes(folder):
    takes = []
    for f in sorted(Path(folder).glob("*.npz")):
        parts = f.stem.split("__")
        d = np.load(f, allow_pickle=False)
        takes.append({"word": parts[0], "person": parts[1] if len(parts) > 1 else "?",
                      "times": d["times"].astype(np.float64),
                      "feats": d["feats"].astype(np.float32)})
    return takes


def run_ctc(model, take, step):
    """Replay one take through the CTC model + streaming decoder."""
    times, feats = take["times"], take["feats"]
    dec, last = StreamingDecoder(), None
    now, end = step, float(times[-1]) + DELAY_SEC + 0.4
    while now <= end:
        n = int(np.searchsorted(times, now, side="right"))
        if n >= 2:
            window = build_features(resample_window(times[:n], feats[:n], now, WINDOW_SEC, T))
            path = predict_paths(model, window[None])[0]
            dec.update(path, now)
            last = (path, now)
        now += step
    if last is not None:
        dec.update(last[0], last[1], final=True)
    return dec.text


def run_old(model, classes, take, step=0.08):
    """Replay one take through the old window classifier + rule-based registrar."""
    from model import predict_probs
    from recognizer import Registrar, SignDecider
    decider = SignDecider(lambda f: predict_probs(model, f), classes)
    registrar = Registrar()
    times, feats = take["times"], take["feats"]
    text, buf_start = "", 0
    now, end = step, float(times[-1]) + 0.5
    while now <= end:
        n = int(np.searchsorted(times, now, side="right"))
        lo = max(buf_start, int(np.searchsorted(times, now - (DYNAMIC_WINDOW_SEC + 0.5))))
        info = decider.decide(times[lo:n], feats[lo:n], now, "ABC")
        label = registrar.update(info["label"], now)
        if label is not None:
            if label == "SPACE":
                text += " "
            elif label == "DELETE":
                text = text[:-1]
            elif len(label) == 1:
                text += label
            if label in DYNAMIC_LABELS:
                buf_start = n          # the live app clears its buffer after J / Z
        now += step
    return text


def summarize(name, refs, hyps):
    exact = float(np.mean([r == h for r, h in zip(refs, hyps)]))
    print(f"  {name:4s}  CER {cer(refs, hyps) * 100:5.1f}%   exact words {exact * 100:5.1f}%")


def main():
    args = parse_args()
    takes = load_word_takes(args.words_dir)
    if not takes:
        raise SystemExit(f"No word takes in '{args.words_dir}'.")
    people = sorted({t["person"] for t in takes})
    print(f"{len(takes)} real word takes, {len({t['word'] for t in takes})} different words, "
          f"signer(s): {people}\n")

    ctc_model, _ = load_ctc(args.ctc)
    results = {"CTC": [run_ctc(ctc_model, t, args.step) for t in takes]}
    if args.old:
        from model import load_checkpoint as load_old
        old_model, classes = load_old(args.old)
        results["OLD"] = [run_old(old_model, classes, t) for t in takes]

    refs = [t["word"] for t in takes]
    print("ALL TAKES")
    for name in ("OLD", "CTC"):
        if name in results:
            summarize(name, refs, results[name])

    fresh = [i for i, w in enumerate(refs) if w not in WORDS]
    if fresh and len(fresh) < len(refs):
        print(f"\nONLY words NOT in the stitched training word list ({len(fresh)} takes)")
        for name in ("OLD", "CTC"):
            if name in results:
                summarize(name, [refs[i] for i in fresh], [results[name][i] for i in fresh])
    elif not fresh:
        print("\nNOTE: every test word is in seq_data.WORDS. If the CTC model was trained before "
              "the word exclusion existed, retrain it (python train_ctc.py) for an honest score.")

    per_word = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(int)
    for i, w in enumerate(refs):
        counts[w] += 1
        for name, hyps in results.items():
            per_word[w][name] += hyps[i] == w
    print("\nPER WORD (takes read exactly right)")
    for w in sorted(counts):
        cells = "   ".join(f"{n} {per_word[w][n]}/{counts[w]}" for n in ("OLD", "CTC") if n in results)
        print(f"  {w:10s} {cells}")

    if args.verbose:
        print("\nWRONG TAKES (truth -> OLD | CTC)")
        for i, w in enumerate(refs):
            if any(results[n][i] != w for n in results):
                print(f"  {w:10s} -> " + " | ".join(f"{n} {results[n][i]!r}" for n in results))


if __name__ == "__main__":
    main()
