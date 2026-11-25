"""Turns a stream of hand frames into confirmed signs (no camera / UI here).

Two windows are used because still and moving signs need different views:
  * short window (1 s)  -> still letters and digits (fast reaction)
  * long window  (2 s)  -> moving signs: J, Z and the words

A sign is only "registered" after it has been predicted several times in a
row, and moving signs must be released before they can fire again. Every
threshold below can be tuned without retraining.
"""
import numpy as np

from utils import (CONTROLS, DYNAMIC_LABELS, DYNAMIC_WINDOW_SEC, IDLE, PRESENCE_COLS,
                   STATIC_WINDOW_SEC, apply_mode, build_features,
                   resample_window, window_motion)

# ---- tunable behaviour ----------------------------------------------------
STATIC_CONF = 0.80          # min probability to accept a still letter/digit
DYNAMIC_CONF = 0.75         # min probability to accept a moving sign
DYNAMIC_SUPPRESS = 0.35     # a moving-sign guess above this blocks still letters
MOTION_GATE = 0.012         # still letters are ignored while the wrist moves faster
STABLE_STATIC = 5           # consecutive predictions needed (about 0.5 s)
STABLE_DYNAMIC = 3          # moving words
REPEAT_SEC = 1.5            # keep holding a still sign this long to type it twice
# J and Z are an I / pointing hand plus movement, so ANY hand movement that
# starts from that shape looks a bit like them. They get stricter settings.
DYNAMIC_LETTERS = {"J", "Z"}
DYNAMIC_LETTER_CONF = 0.90
STABLE_DYNAMIC_LETTER = 6
# Control signs: SPACE fires once per gesture (lower the hands to re-arm),
# DELETE repeats while the fists stay up.
CONTROL_REPEAT_SEC = {"SPACE": 1e9, "DELETE": 0.8}
DYNAMIC_LOCKOUT_SEC = 0.8   # pause after a moving sign (its tail is still in the window)
HAND_LOST_SEC = 0.3         # no hand for this long -> nothing is predicted


class SignDecider:
    """Runs the network on the live buffer and picks at most one sign."""

    def __init__(self, predict_fn, classes):
        self.predict_fn = predict_fn  # (N, T, F) array -> (N, C) probabilities
        self.classes = list(classes)

    def decide(self, times, feats, now: float, mode: str) -> dict:
        info = {"label": None, "conf": 0.0, "static": ("-", 0.0),
                "dynamic": ("-", 0.0), "motion": 0.0}
        if mode == "PAUSE":
            return info
        times = np.asarray(times, dtype=np.float64)
        feats = np.asarray(feats, dtype=np.float32)
        recent = times >= now - HAND_LOST_SEC
        if len(times) < 5 or not recent.any() or not feats[recent][:, PRESENCE_COLS].any():
            return info

        short = build_features(resample_window(times, feats, now, STATIC_WINDOW_SEC))
        long_ = build_features(resample_window(times, feats, now, DYNAMIC_WINDOW_SEC))
        probs = self.predict_fn(np.stack([short, long_]))
        p_short = apply_mode(probs[0], self.classes, mode)
        p_long = apply_mode(probs[1], self.classes, mode)

        dyn = {k: v for k, v in p_long.items() if k in DYNAMIC_LABELS}
        sta = {k: v for k, v in p_short.items()
               if k not in DYNAMIC_LABELS and k != IDLE}
        dyn_label, dyn_conf = max(dyn.items(), key=lambda kv: kv[1], default=("-", 0.0))
        sta_label, sta_conf = max(sta.items(), key=lambda kv: kv[1], default=("-", 0.0))
        motion = window_motion(short)
        # A word signed in the wrong mode must not leak out as a random letter,
        # so still letters are blocked by ANY moving-sign guess, allowed or not.
        raw_long = dict(zip(self.classes, probs[1]))
        dyn_any = max((float(v) for k, v in raw_long.items() if k in DYNAMIC_LABELS),
                      default=0.0)
        info.update(static=(sta_label, sta_conf), dynamic=(dyn_label, dyn_conf),
                    motion=motion)

        need_conf = DYNAMIC_LETTER_CONF if dyn_label in DYNAMIC_LETTERS else DYNAMIC_CONF
        if dyn_conf >= need_conf:
            info.update(label=dyn_label, conf=dyn_conf)
        elif (sta_conf >= STATIC_CONF and motion <= MOTION_GATE
              and dyn_any < DYNAMIC_SUPPRESS):
            info.update(label=sta_label, conf=sta_conf)
        return info


class Registrar:
    """Decides when a prediction becomes a typed character or word."""

    def __init__(self):
        self.candidate = None
        self.count = 0
        self.last_label = None
        self.last_time = -1e9
        self.blocked = None       # moving sign that must be released first
        self.quiet = 0
        self.lockout_until = 0.0

    @staticmethod
    def needed(label) -> int:
        if label in DYNAMIC_LETTERS:
            return STABLE_DYNAMIC_LETTER
        return STABLE_DYNAMIC if label in DYNAMIC_LABELS else STABLE_STATIC

    @property
    def progress(self) -> float:
        if self.candidate is None:
            return 0.0
        return min(1.0, self.count / self.needed(self.candidate))

    def update(self, label, now: float):
        """Feed one prediction (or None). Returns a label when one registers."""
        if now < self.lockout_until:
            self.candidate, self.count = None, 0
            return None
        if label is None:
            self.candidate, self.count = None, 0
            self.quiet += 1
            if self.quiet >= 3:
                self.blocked = None
                if self.last_label in CONTROLS:   # hands lowered: re-arm the control sign
                    self.last_label = None
            return None
        self.quiet = 0
        if self.blocked is not None:
            if label == self.blocked:
                return None
            self.blocked = None
        if label != self.candidate:
            self.candidate, self.count = label, 1
        else:
            self.count += 1
        if self.count < self.needed(label):
            return None

        dynamic = label in DYNAMIC_LABELS
        if (not dynamic and label == self.last_label
                and now - self.last_time < CONTROL_REPEAT_SEC.get(label, REPEAT_SEC)):
            return None  # still holding the same letter
        self.last_label, self.last_time = label, now
        if dynamic:
            self.blocked = label
            self.lockout_until = now + DYNAMIC_LOCKOUT_SEC
            self.candidate, self.count = None, 0
        return label
