"""Shared constants and helpers for the ASL sign recogniser.

Everything that must be IDENTICAL in data collection, training and live
recognition lives here (labels, feature extraction, windowing), so the three
scripts can never drift apart.

Data flow
---------
camera frame -> MediaPipe hand landmarks -> frame vector (132 numbers)
-> time window resampled to SEQ_LEN frames -> feature matrix (SEQ_LEN x 136)
-> neural network -> label probabilities
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")
WORDS = ["HELLO", "GOODBYE", "THANK_YOU", "YES", "NO", "ME", "YOU", "NAME", "HOW_ARE_YOU"]
# Control signs act like keyboard keys. They use BOTH hands (every letter and
# digit uses one), so they cannot be mistaken for a letter.
#   SPACE  = both open palms up, side by side      DELETE = both fists up
CONTROLS = ["SPACE", "DELETE"]
IDLE = "IDLE"  # hand visible but not signing (resting / moving between signs)
ALL_LABELS = LETTERS + DIGITS + WORDS + CONTROLS + [IDLE]

# Signs defined by movement (J and Z are drawn in the air).
DYNAMIC_LABELS = {"J", "Z"} | set(WORDS)

# Digit -> letter with the same (or nearly the same) handshape in ASL.
# The model cannot tell these apart from one frame, so at recognition time
# the chosen mode decides which one is meant (see apply_mode).
TWINS = {"0": "O", "1": "D", "2": "V", "6": "W", "9": "F"}

MODES = ["ABC", "123", "WORDS", "PAUSE"]


def canonical(label: str) -> str:
    """Map a digit onto its look-alike letter (used when scoring accuracy)."""
    return TWINS.get(label, label)


def allowed_labels(mode: str) -> set:
    """Labels that may be output in each mode. The content modes never mix:
    ABC = letters only, 123 = digits only, WORDS = words only.
    The control signs (SPACE, DELETE) work in all three. PAUSE ignores everything."""
    if mode == "ABC":
        return set(LETTERS) | set(CONTROLS) | {IDLE}
    if mode == "123":
        return set(DIGITS) | set(CONTROLS) | {IDLE}
    if mode == "WORDS":
        return set(WORDS) | set(CONTROLS) | {IDLE}
    return set()  # "PAUSE"


def apply_mode(probs, classes, mode: str) -> dict:
    """Turn raw class probabilities into {label: prob} for the active mode.

    Look-alike pairs (0/O, 2/V ...) are merged onto whichever member the mode
    allows, so their probability mass is not split and lost.
    """
    p = {c: float(v) for c, v in zip(classes, probs)}
    for digit, letter in TWINS.items():
        if digit in p and letter in p:
            target = digit if mode == "123" else letter
            total = p[digit] + p[letter]
            p[digit] = p[letter] = 0.0
            p[target] = total
    allowed = allowed_labels(mode)
    return {c: p[c] for c in classes if c in allowed}


# --------------------------------------------------------------------------
# Windowing constants
# --------------------------------------------------------------------------
SEQ_LEN = 30                  # frames fed to the network
STATIC_WINDOW_SEC = 1.0       # live window used for still letters/digits
DYNAMIC_WINDOW_SEC = 2.0      # live window used for moving signs
STATIC_TRAIN_RANGE = (0.8, 1.3)    # random window length while training
DYNAMIC_TRAIN_RANGE = (1.6, 2.4)   # (speed augmentation comes for free)
IDLE_TRAIN_RANGE = (0.8, 2.4)
TAKE_SEC = 3.0                # length of one recorded take
DYNAMIC_ZONE = (0.8, 2.2)     # when the sign is performed inside a take
MAX_GAP_SEC = 0.2             # a gap longer than this counts as "no data"

# --------------------------------------------------------------------------
# Per-frame vector layout
#   two hand slots x [63 shape values, wrist x, wrist y, present flag]
# --------------------------------------------------------------------------
SLOT_DIM = 66
NUM_SLOTS = 2
RAW_DIM = SLOT_DIM * NUM_SLOTS      # 132
FEAT_DIM = RAW_DIM + 4              # + wrist velocity of both slots = 136
PRESENCE_COLS = [SLOT_DIM * s + 65 for s in range(NUM_SLOTS)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def hand_to_slot(landmarks, handedness: str) -> np.ndarray:
    """One detected hand -> 66 numbers.

    shape (63): landmarks relative to the wrist, divided by the wrist-to-
    middle-finger-base distance (distance invariant). A left hand is mirrored
    so both hands look like a right hand to the model.
    position (2): where the wrist is in the image. The shape alone cannot tell
    HELLO (hand at the forehead) from a still B, or J from I; position can.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist_xy = coords[0, :2].copy()
    rel = coords - coords[0]
    if handedness == "Left":
        rel[:, 0] *= -1.0
    scale = float(np.linalg.norm(rel[9]))
    if scale > 1e-6:
        rel /= scale
    slot = np.zeros(SLOT_DIM, dtype=np.float32)
    slot[:63] = rel.ravel()
    slot[63:65] = wrist_xy
    slot[65] = 1.0
    return slot


def frame_features(result) -> np.ndarray:
    """MediaPipe result -> 132-number frame vector (zeros if no hand)."""
    slots = []
    for landmarks, handed in zip(result.hand_landmarks, result.handedness):
        slots.append(hand_to_slot(landmarks, handed[0].category_name))
    slots.sort(key=lambda s: s[63])  # left-most hand in the image first
    vec = np.zeros(RAW_DIM, dtype=np.float32)
    for i, slot in enumerate(slots[:NUM_SLOTS]):
        vec[i * SLOT_DIM:(i + 1) * SLOT_DIM] = slot
    return vec


def resample_window(times, feats, t_end: float, duration: float,
                    n: int = SEQ_LEN) -> np.ndarray:
    """Pick n frames evenly spaced over [t_end - duration, t_end].

    Sampling by TIME (nearest frame) rather than by frame count makes the
    window independent of the camera/computer frame rate. Moments with no
    data (before recording started, or a long gap) become all-zero frames.
    """
    times = np.asarray(times, dtype=np.float64)
    feats = np.asarray(feats, dtype=np.float32)
    if len(times) < 2:
        return np.zeros((n, RAW_DIM), dtype=np.float32)
    targets = np.linspace(t_end - duration, t_end, n)
    idx = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    left, right = times[idx - 1], times[idx]
    pick = np.where(targets - left <= right - targets, idx - 1, idx)
    out = feats[pick].copy()
    gap = np.abs(times[pick] - targets) > MAX_GAP_SEC
    out[gap] = 0.0
    return out


def build_features(raw: np.ndarray) -> np.ndarray:
    """(T, 132) raw window -> (T, 136) network input (adds wrist velocity)."""
    pos = raw[:, [63, 64, SLOT_DIM + 63, SLOT_DIM + 64]]
    pres = raw[:, PRESENCE_COLS]
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    both = np.minimum(pres[1:], pres[:-1])            # (T-1, 2)
    vel[1:] *= np.repeat(both, 2, axis=1)             # ignore missing frames
    return np.concatenate([raw, vel], axis=1).astype(np.float32)


def window_motion(feats: np.ndarray) -> float:
    """Average wrist speed in a built feature window (used to gate letters)."""
    vel = feats[:, RAW_DIM:RAW_DIM + 4].reshape(-1, 2, 2)
    speed = np.linalg.norm(vel, axis=2)               # (T, 2 hands)
    return float(speed.max(axis=1).mean())


# --------------------------------------------------------------------------
# Training-window sampling and augmentation
# --------------------------------------------------------------------------
def sample_window(times, feats, label: str, rng, duration=None) -> np.ndarray:
    """Cut one training window out of a recorded take.

    Dynamic signs are performed inside DYNAMIC_ZONE of the take, so their
    windows are forced to contain the whole zone (the sign is never cut off).
    """
    if duration is None:
        if label == IDLE:
            lo_d, hi_d = IDLE_TRAIN_RANGE
        elif label in DYNAMIC_LABELS:
            lo_d, hi_d = DYNAMIC_TRAIN_RANGE
        else:
            lo_d, hi_d = STATIC_TRAIN_RANGE
        duration = rng.uniform(lo_d, hi_d)
    length = float(times[-1])
    duration = min(duration, length)
    latest = max(length - duration, 0.0)
    lo, hi = 0.0, latest
    if label in DYNAMIC_LABELS:
        lo = max(0.0, DYNAMIC_ZONE[1] - duration)
        hi = min(DYNAMIC_ZONE[0], latest)
        if lo > hi:
            lo, hi = 0.0, latest
    start = rng.uniform(lo, hi) if hi > lo else lo
    return resample_window(times, feats, start + duration, duration)


def _rotation_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = (np.cos(ax), np.sin(ax), np.cos(ay),
                              np.sin(ay), np.cos(az), np.sin(az))
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def augment_window(raw: np.ndarray, rng) -> np.ndarray:
    """Random rotation / scale / noise / shift / mirror / dropped frames."""
    r = raw.reshape(len(raw), NUM_SLOTS, SLOT_DIM).copy()
    rot = _rotation_matrix(*rng.normal(0.0, [0.10, 0.10, 0.15]))
    scale = rng.uniform(0.92, 1.08)
    shift = rng.normal(0.0, 0.05, 2)
    for s in range(NUM_SLOTS):
        present = r[:, s, 65]
        shape = r[:, s, :63].reshape(-1, 21, 3)
        shape = (shape @ rot.T) * scale + rng.normal(0.0, 0.015, shape.shape)
        shape *= present[:, None, None]
        r[:, s, :63] = shape.reshape(-1, 63)
        r[:, s, 63:65] += shift * present[:, None]
    if rng.random() < 0.5:  # mirror the whole scene (left-handed signers)
        r[:, :, 63] = np.where(r[:, :, 65] > 0, 1.0 - r[:, :, 63], 0.0)
        if r[:, 1, 65].any():
            r = r[:, ::-1].copy()
    if rng.random() < 0.3:  # simulate the hand tracker losing a few frames
        for _ in range(int(rng.integers(1, 4))):
            r[int(rng.integers(0, len(r)))] = 0.0
    return r.reshape(len(raw), RAW_DIM)


# --------------------------------------------------------------------------
# Camera + MediaPipe helpers (imported lazily so the rest works without them)
# --------------------------------------------------------------------------
def open_camera(index: int = 0, width: int = 640, height: int = 480):
    """Open the webcam. 640x480 keeps the frame rate high on laptop CPUs."""
    import cv2
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}. Try --camera 1.")
    return cap


class HandTracker:
    """Thin wrapper around MediaPipe HandLandmarker (VIDEO mode, 2 hands)."""

    def __init__(self, model_path: str = "hand_landmarker.task"):
        import mediapipe as mp
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"'{model_path}' not found. Download it (see README step 1) "
                "and put it next to the scripts.")
        self._mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1

    def detect(self, frame_bgr):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self._t0) * 1000)
        ts = max(ts, self._last_ts + 1)  # timestamps must strictly increase
        self._last_ts = ts
        return self._landmarker.detect_for_video(image, ts)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_hands(frame, result) -> None:
    """Draw the detected hand skeletons onto the frame."""
    import cv2
    h, w = frame.shape[:2]
    for landmarks in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)
