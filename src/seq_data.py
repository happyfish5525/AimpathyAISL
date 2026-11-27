"""Sequence data for the CTC fingerspelling model (numpy only, no torch).

Why this file exists
--------------------
Every recorded take is ONE isolated sign, but the CTC model must learn to read
several letters in a row. So we build training sequences by stitching isolated
takes together:

    [idle/no hand] letter1 -blend- letter2 -blend- letter3 ... [idle/no hand]

* A letter segment is cut from a real take of that letter (J and Z use their
  moving part).
* Segments are moved to roughly the same hand position, like one signer
  spelling in one spot.
* Between two letters the hand shape is blended over 0.08-0.3 s (a quick
  handshape change).
* Between two IDENTICAL letters (LL in HELLO) the hand does a small bounce, so
  the model can learn that a repeat needs a gap. ASSUMPTION: real signers do
  something like this; the real recorded words (collect_words.py, next step)
  are what will confirm or correct it.
* The whole sequence then gets the same augmentation as before (rotation,
  scale, noise, mirroring, dropped frames) and is turned into the same
  136-number-per-frame features as the old model, on a fixed 25 fps grid.

Vocabulary: blank (index 0), A-Z, SPACE (both open palms). Digits and the word
signs stay with the old window classifier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils import (DYNAMIC_ZONE, IDLE, LETTERS, NUM_SLOTS, PRESENCE_COLS,
                   RAW_DIM, SLOT_DIM, augment_window, build_features,
                   resample_window)

FPS = 25                         # fixed frame rate the CTC model sees
WINDOW_SEC = 4.0                 # length of one training / live window
T = int(FPS * WINDOW_SEC)        # 100 frames
VOCAB = ["<blank>"] + LETTERS + ["SPACE"]
TOKEN_TO_IDX = {t: i for i, t in enumerate(VOCAB)}
MOVING_LETTERS = {"J", "Z"}      # drawn in the air

WORDS = """HELLO COFFEE BOOK MOON SEE ALL BALL LITTLE SUMMER HAPPY APPLE GOOD FOOD
TOOL SCHOOL WEEK TREE BEE GREEN SORRY PLEASE THANK FRIEND FAMILY WATER HOUSE
MOTHER FATHER SISTER BROTHER TEACHER STUDENT DOCTOR NURSE HOSPITAL MARKET
RICE TEA MILK BREAD EGG FISH CHICKEN BEEF SUGAR SALT PIZZA BURGER JUICE
CAT DOG BIRD HORSE TIGER LION MONKEY RABBIT ELEPHANT
RED BLUE YELLOW BLACK WHITE PINK ORANGE PURPLE
ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN
JOHN MARY AMIR SARAH ALI MEI ZARA OMAR LILY DAVID EMMA NOOR
JOHOR PENANG MALAYSIA ASIA EARTH SUN STAR RAIN WIND FIRE
QUIZ JAZZ ZEBRA JUMP QUICK EXAM WORK PLAY LOVE HOPE HELP STOP GO COME
YES NO NAME TODAY TOMORROW MORNING NIGHT""".split()

ACTIVE_WORDS = list(WORDS)      # the words stitched sequences are drawn from


def exclude_words(words) -> int:
    """Remove words from the stitched-training vocabulary.

    Call this with the words of the real TEST recordings so the model can
    never have seen them (not even as stitched sequences). Returns how many
    words were removed.
    """
    global ACTIVE_WORDS
    banned = {w.upper() for w in words}
    ACTIVE_WORDS = [w for w in WORDS if w not in banned]
    return len(WORDS) - len(ACTIVE_WORDS)


# --------------------------------------------------------------------------
# Loading takes
# --------------------------------------------------------------------------
def load_takes(folder):
    """All takes in a folder whose label is A-Z, SPACE or IDLE."""
    takes = []
    for f in sorted(Path(folder).glob("*.npz")):
        parts = f.stem.split("__")
        label = parts[0]
        if label == IDLE or (label in TOKEN_TO_IDX and label != "<blank>"):
            d = np.load(f, allow_pickle=False)
            takes.append({"id": f.stem, "label": label,
                          "person": parts[1] if len(parts) > 1 else "?",
                          "times": d["times"].astype(np.float64),
                          "feats": d["feats"].astype(np.float32)})
    return takes


def make_bank(takes):
    """{label: [takes]}"""
    bank = {}
    for t in takes:
        bank.setdefault(t["label"], []).append(t)
    return bank


def split_takes(takes, rng, val_frac=0.15):
    """Hold out whole takes per label (labels with < 4 takes are not split)."""
    by_label = {}
    for t in takes:
        by_label.setdefault(t["label"], []).append(t)
    train, val = [], []
    for group in by_label.values():
        order = rng.permutation(len(group))
        n_val = max(1, round(len(group) * val_frac)) if len(group) >= 4 else 0
        val += [group[i] for i in order[:n_val]]
        train += [group[i] for i in order[n_val:]]
    return train, val


# --------------------------------------------------------------------------
# Building one sequence
# --------------------------------------------------------------------------
def hand_fraction(raw) -> float:
    return float((raw[:, PRESENCE_COLS].max(axis=1) > 0).mean())


HOLD_RANGES = [(0.20, 0.50), (0.40, 0.90), (0.70, 1.50)]   # fast / normal / slow signers


def cut_segment(take, rng, hold=(0.22, 0.6)) -> np.ndarray:
    """Cut the part of a take that shows its sign, on the 25 fps grid."""
    label, times = take["label"], take["times"]
    length = float(times[-1])
    seg = None
    for _ in range(6):                       # retry if the hand was mostly missing
        if label in MOVING_LETTERS:
            start = max(0.0, DYNAMIC_ZONE[0] + rng.uniform(-0.1, 0.1))
            end = min(length, DYNAMIC_ZONE[1] + rng.uniform(-0.1, 0.1))
        else:
            lo, hi = (0.5, 1.0) if label == "SPACE" else hold
            dur = min(rng.uniform(lo, hi), max(length - 0.4, 0.3))
            start = rng.uniform(0.3, max(0.3, length - 0.1 - dur))
            end = start + dur
        n = max(3, int(round((end - start) * FPS)))
        seg = resample_window(times, take["feats"], end, end - start, n)
        if hand_fraction(seg) >= 0.8:
            break
    return seg


def shift_to(seg, anchor):
    """Move the wrist track of hand slot 0 so its mean sits at `anchor`."""
    present = seg[:, 65] > 0
    if not present.any():
        return seg
    seg = seg.copy()
    cur = seg[present][:, 63:65].mean(axis=0)
    seg[present, 63:65] += anchor - cur
    return seg


def transition(a_last, b_first, n, bounce, rng) -> np.ndarray:
    """n frames that morph one hand pose into the next (optionally a bounce)."""
    w = np.linspace(0.0, 1.0, n + 2)[1:-1]
    out = np.zeros((n, RAW_DIM), dtype=np.float32)
    for s in range(NUM_SLOTS):
        sl = slice(s * SLOT_DIM, (s + 1) * SLOT_DIM)
        a, b = a_last[sl], b_first[sl]
        if a[65] > 0 and b[65] > 0:                  # both hands present: blend
            out[:, sl] = (1 - w)[:, None] * a + w[:, None] * b
        else:                                        # appear / disappear: step
            out[:, sl] = np.where((w < 0.5)[:, None], a, b)
    if bounce:                                       # small dip of the wrist
        amp = rng.uniform(0.02, 0.06)
        out[:, 64] += amp * np.sin(np.pi * w) * (out[:, 65] > 0)
    return out


def filler(bank, n, rng) -> np.ndarray:
    """n frames of 'nobody is signing': an IDLE take part or no hand at all."""
    if n <= 0:
        return np.zeros((0, RAW_DIM), dtype=np.float32)
    idle = bank.get(IDLE)
    if idle and rng.random() < 0.5:
        take = idle[int(rng.integers(len(idle)))]
        length, dur = float(take["times"][-1]), n / FPS
        end = rng.uniform(min(dur, length), length)
        return resample_window(take["times"], take["feats"], end, dur, n)
    return np.zeros((n, RAW_DIM), dtype=np.float32)


def build_sequence(bank, tokens, rng):
    """Stitch `tokens` into one (T, 132) raw window.

    Returns (raw, kept_tokens). Tokens that do not fit in the window, or that
    have no recorded take, are dropped from the END, so the target always
    matches what is really in the window.
    """
    pieces, kept = [], []
    used, anchor, prev_seg, prev_tok = 0, None, None, None
    budget = T - 4
    # one signing speed per sequence: real people hold each letter 0.3-1.5 s
    hold = HOLD_RANGES[int(rng.integers(len(HOLD_RANGES)))]
    for tok in tokens:
        if not bank.get(tok):
            continue
        take = bank[tok][int(rng.integers(len(bank[tok])))]
        seg = cut_segment(take, rng, hold)
        if tok != "SPACE":
            if anchor is None:
                present = seg[:, 65] > 0
                anchor = (seg[present][:, 63:65].mean(axis=0) if present.any()
                          else np.array([0.5, 0.55], dtype=np.float32))
            else:
                anchor = np.clip(anchor + rng.normal(0.0, 0.01, 2), 0.1, 0.9)
            seg = shift_to(seg, anchor)
        if prev_seg is None:
            gap = np.zeros((0, RAW_DIM), dtype=np.float32)
        else:
            same = tok == prev_tok
            lo, hi = (0.15, 0.30) if same else (0.06, 0.22)
            n = max(2, int(round(rng.uniform(lo, hi) * FPS)))
            gap = transition(prev_seg[-1], seg[0], n, same, rng)
        if used + len(gap) + len(seg) > budget:
            break
        pieces += [gap, seg]
        used += len(gap) + len(seg)
        kept.append(tok)
        prev_seg, prev_tok = seg, tok

    remaining = T - used
    lead = int(rng.beta(1.0, 2.0) * remaining)       # usually a short lead-in
    body = np.concatenate(pieces) if pieces else np.zeros((0, RAW_DIM), np.float32)
    raw = np.concatenate([filler(bank, lead, rng), body,
                          filler(bank, remaining - lead, rng)]).astype(np.float32)
    if len(raw) < T:
        raw = np.concatenate([raw, np.zeros((T - len(raw), RAW_DIM), np.float32)])
    return raw[:T], kept


# --------------------------------------------------------------------------
# Real recorded words (collect_words.py) as extra training data
# --------------------------------------------------------------------------
MAX_SQUEEZE = 1.5        # a word may be sped up at most this much to fit the window
MOTION_START = 0.15      # smoothed hand-shape change per frame that counts as "signing started"


def load_word_takes(folder):
    """Real word takes: data/words_train/HELLO__lls__1795230000000.npz -> label 'HELLO'."""
    takes = []
    for f in sorted(Path(folder).glob("*.npz")):
        parts = f.stem.split("__")
        word = parts[0].upper()
        if not word or any(c not in LETTERS for c in word):
            continue
        d = np.load(f, allow_pickle=False)
        takes.append({"id": f.stem, "label": word,
                      "person": parts[1] if len(parts) > 1 else "?",
                      "times": d["times"].astype(np.float64),
                      "feats": d["feats"].astype(np.float32)})
    return takes


def real_word_window(take, rng, speed=None):
    """One real word take -> a (T, 132) raw window on the 25 fps grid, or None.

    The idle lead-in is trimmed. A word longer than the 4 s window is sped up
    (at most MAX_SQUEEZE times); if it would need more, it is skipped, because
    cutting a word would leave letters in the label that are not in the window.
    """
    times, feats = take["times"], take["feats"]
    if len(times) < 10:
        return None
    length = float(times[-1])
    hand = feats[:, :63]
    v = np.r_[0.0, np.linalg.norm(np.diff(hand, axis=0), axis=1)]
    v = np.convolve(v, np.ones(5) / 5.0, mode="same")
    moving = np.flatnonzero(v > MOTION_START)
    start = max(0.0, float(times[moving[0]]) - 0.3) if len(moving) else 0.0
    seg_dur = length - start
    fastest_ok = seg_dur / (WINDOW_SEC - 0.1)          # needed to fit the window
    factor = max(fastest_ok, 1.0 if speed is None else speed)
    if factor > MAX_SQUEEZE or seg_dur < 0.8:
        return None
    n = min(T, max(8, int(round(seg_dur / factor * FPS))))
    seg = resample_window(times, feats, length, seg_dur, n)
    if hand_fraction(seg) < 0.5:
        return None
    extra = T - n
    lead = int(rng.uniform(0.0, 0.5) * extra)
    raw = np.concatenate([np.repeat(seg[:1], lead, axis=0), seg,
                          np.repeat(seg[-1:], extra - lead, axis=0)])
    return raw.astype(np.float32)


def extra_roll(raw, rng, max_rad=0.45):
    """Turn the whole hand shape in the image plane by a random angle (+-26 deg).

    Different people hold the same letter at slightly different wrist angles
    (measured: the P, G and Y of two signers differ by 15-25 deg), and the
    standard augmentation only turns by about 6 deg.
    """
    from utils import SLOT_DIM, NUM_SLOTS
    theta = rng.uniform(-max_rad, max_rad)
    c, s_ = np.cos(theta), np.sin(theta)
    out = raw.copy()
    for k in range(NUM_SLOTS):
        sl = slice(k * SLOT_DIM, k * SLOT_DIM + 63)
        shape = out[:, sl].reshape(len(out), 21, 3)
        x, y = shape[..., 0].copy(), shape[..., 1].copy()
        shape[..., 0] = c * x - s_ * y
        shape[..., 1] = s_ * x + c * y
        out[:, sl] = shape.reshape(len(out), 63)
    return out


def sample_real_example(word_takes, rng, augment=True):
    """-> (features (T, 136), target token indices) or None."""
    for _ in range(10):
        take = word_takes[int(rng.integers(len(word_takes)))]
        raw = real_word_window(take, rng, speed=rng.uniform(0.9, 1.15))
        if raw is not None:
            if augment:
                raw = extra_roll(augment_window(raw, rng), rng)
            return build_features(raw), [TOKEN_TO_IDX[c] for c in take["label"]]
    return None


def make_real_set(word_takes, seed=0):
    """One un-augmented example per usable take (for validation)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for take in word_takes:
        raw = real_word_window(take, rng, speed=1.0)
        if raw is not None:
            xs.append(build_features(raw))
            ys.append([TOKEN_TO_IDX[c] for c in take["label"]])
    if not xs:
        return None, []
    return np.stack(xs).astype(np.float32), ys


# --------------------------------------------------------------------------
# Sampling examples
# --------------------------------------------------------------------------
def random_tokens(rng):
    """Half real words, some two-word phrases, the rest random letter strings."""
    r = rng.random()
    if r < 0.50:
        return list(ACTIVE_WORDS[int(rng.integers(len(ACTIVE_WORDS)))])
    if r < 0.65:
        a, b = (ACTIVE_WORDS[int(rng.integers(len(ACTIVE_WORDS)))] for _ in range(2))
        return list(a) + ["SPACE"] + list(b)
    n = int(rng.integers(2, 7))
    toks = [LETTERS[int(rng.integers(26))] for _ in range(n)]
    if n >= 3 and rng.random() < 0.25:               # force a double letter
        i = int(rng.integers(1, n))
        toks[i] = toks[i - 1]
    return toks


def sample_example(bank, rng, augment=True):
    """-> (features (T, 136), target token indices)"""
    raw, kept = build_sequence(bank, random_tokens(rng), rng)
    if augment:
        raw = extra_roll(augment_window(raw, rng), rng)
    return build_features(raw), [TOKEN_TO_IDX[t] for t in kept]


def make_set(bank, n, seed, augment=True, word_takes=None, real_frac=0.0):
    """n examples: stitched from single letters, plus (real_frac of them) real words."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        ex = None
        if word_takes and rng.random() < real_frac:
            ex = sample_real_example(word_takes, rng, augment)
        if ex is None:
            ex = sample_example(bank, rng, augment)
        xs.append(ex[0])
        ys.append(ex[1])
    return np.stack(xs).astype(np.float32), ys


# --------------------------------------------------------------------------
# CTC decoding helpers and the error metric
# --------------------------------------------------------------------------
def collapse_path(path):
    """Greedy CTC collapse of a per-frame argmax path.

    Returns [(centre_frame, token_index), ...]: each run of the same
    non-blank token is one letter, positioned at the middle of the run.
    A repeated letter therefore needs a blank frame between its two runs.
    """
    out, i, n = [], 0, len(path)
    while i < n:
        j = i
        while j + 1 < n and path[j + 1] == path[i]:
            j += 1
        if path[i] != 0:
            out.append(((i + j) / 2.0, int(path[i])))
        i = j + 1
    return out


def tokens_to_text(indices) -> str:
    return "".join(" " if VOCAB[i] == "SPACE" else VOCAB[i] for i in indices)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(refs, hyps) -> float:
    """Character error rate = total edit distance / total reference length."""
    total = sum(len(r) for r in refs)
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / max(total, 1)