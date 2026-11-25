"""Record training data as "takes" (a few seconds of hand frames each).

Why takes? The old collector saved a new overlapping window on every key
repeat, so 78% of rows were near-copies of each other and the test score was
meaningless. Here each take is a separate file, and training splits by take.

Usage
-----
    python collect_data.py                 # training data  -> data/takes
    python collect_data.py --person Sam    # tag who is signing
    python collect_data.py --out data/test # separate test set (different day/light!)

Keys
----
    A-Z, 0-9      choose the label        TAB or ]   next label
    -             choose IDLE             [          previous label
    SPACE         record a burst of takes BACKSPACE  cancel / delete last take
    ESC           quit
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from utils import (ALL_LABELS, CONTROLS, DYNAMIC_LABELS, DYNAMIC_ZONE, IDLE,
                   PRESENCE_COLS, TAKE_SEC, HandTracker, draw_hands,
                   frame_features, open_camera)

WAIT_SEC = 1.5          # get-ready time before every take
MIN_VISIBLE = 0.70      # discard takes where the hand was visible less than this
STATIC_TARGET = 10      # recommended takes per still sign
DYNAMIC_TARGET = 20     # recommended takes per moving sign
CONTROL_HINTS = {
    "SPACE": "HOLD BOTH hands up, open palms facing the camera, side by side",
    "DELETE": "HOLD BOTH hands up as fists, side by side",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/takes", help="folder for the takes")
    p.add_argument("--person", default="me", help="name tag stored with each take")
    p.add_argument("--burst", type=int, default=5, help="takes recorded per SPACE press")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--landmarker", default="hand_landmarker.task")
    return p.parse_args()


def count_takes(folder: Path) -> dict:
    counts = {}
    for f in folder.glob("*.npz"):
        label = f.name.split("__")[0]
        counts[label] = counts.get(label, 0) + 1
    return counts


def target_for(label: str) -> int:
    return DYNAMIC_TARGET if label in DYNAMIC_LABELS else STATIC_TARGET


def save_take(folder: Path, label: str, person: str, times, feats):
    """Validate and store one take. Returns (path or None, message)."""
    feats = np.stack(feats).astype(np.float32)
    times = np.asarray(times, dtype=np.float32)
    shown = feats[:, PRESENCE_COLS]               # control signs need BOTH hands visible
    visible = float((shown.all(axis=1) if label in CONTROLS else shown.any(axis=1)).mean())
    fps = len(times) / TAKE_SEC
    if label != IDLE and visible < MIN_VISIBLE:
        return None, f"Discarded: hand visible only {visible:.0%} of the take"
    path = folder / f"{label}__{person}__{int(time.time() * 1000)}.npz"
    np.savez_compressed(path, times=times, feats=feats, label=label, person=person)
    warn = "  (LOW FPS - close other programs)" if fps < 10 else ""
    return path, f"Saved ({visible:.0%} visible, {fps:.0f} fps){warn}"


def put(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    args = parse_args()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    counts = count_takes(folder)

    label_idx = 0
    state = "idle"            # idle -> wait -> record -> (wait -> record ...) -> idle
    phase_start = 0.0
    takes_left = 0
    times, feats = [], []
    last_saved = None
    message, message_until = "", 0.0

    print(__doc__)
    tracker = HandTracker(args.landmarker)
    cap = open_camera(args.camera)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            now = time.time()
            result = tracker.detect(frame)
            vec = frame_features(result)
            draw_hands(frame, result)
            label = ALL_LABELS[label_idx]
            dynamic = label in DYNAMIC_LABELS

            # ---- state machine ------------------------------------------
            if state == "wait" and now - phase_start >= WAIT_SEC:
                state, phase_start, times, feats = "record", now, [], []
            elif state == "record":
                times.append(now - phase_start)
                feats.append(vec)
                if now - phase_start >= TAKE_SEC:
                    path, message = save_take(folder, label, args.person, times, feats)
                    message_until = now + 3
                    if path is not None:
                        last_saved = path
                        counts[label] = counts.get(label, 0) + 1
                    takes_left -= 1
                    state = "wait" if takes_left > 0 else "idle"
                    phase_start = now

            # ---- overlay ------------------------------------------------
            n = counts.get(label, 0)
            goal = target_for(label)
            kind = "MOVING sign" if dynamic else "STILL sign"
            put(frame, f"Label: {label}  [{kind}]", 28, (255, 255, 0), 0.7)
            put(frame, f"Takes: {n}/{goal}   person: {args.person}", 56,
                (0, 255, 0) if n >= goal else (0, 165, 255))
            if state == "idle":
                put(frame, "SPACE = record burst   TAB = next label", 84, (200, 200, 200), 0.5)
            elif state == "wait":
                hint = ("Move your hand a little closer/farther/rotated"
                        if takes_left < args.burst else "Get ready")
                put(frame, f"{hint}  ({takes_left} left)", 84, (0, 255, 255))
            else:
                t = now - phase_start
                in_zone = DYNAMIC_ZONE[0] <= t <= DYNAMIC_ZONE[1]
                if dynamic:
                    put(frame, "SIGN NOW!" if in_zone else "wait...", 84,
                        (0, 255, 0) if in_zone else (0, 165, 255), 0.8)
                else:
                    put(frame, CONTROL_HINTS.get(
                        label, "HOLD the sign (small natural movement is fine)"), 84,
                        (0, 255, 0))
                w = frame.shape[1]
                h = frame.shape[0]
                cv2.rectangle(frame, (0, h - 14), (int(w * t / TAKE_SEC), h), (0, 0, 255), -1)
                if dynamic:
                    z0, z1 = int(w * DYNAMIC_ZONE[0] / TAKE_SEC), int(w * DYNAMIC_ZONE[1] / TAKE_SEC)
                    cv2.rectangle(frame, (z0, h - 22), (z1, h - 16), (0, 255, 0), -1)
            if now < message_until:
                put(frame, message, frame.shape[0] - 40, (0, 255, 255), 0.55)

            cv2.imshow("ASL data collector", frame)
            key = cv2.waitKey(1) & 0xFF

            # ---- keys ---------------------------------------------------
            if key == 27:
                break
            if key == 32 and state == "idle":
                state, phase_start, takes_left = "wait", now, args.burst
            elif key == 8:
                if state != "idle":
                    state, takes_left = "idle", 0
                    message, message_until = "Cancelled", now + 2
                elif last_saved is not None and last_saved.exists():
                    last_saved.unlink()
                    lab = last_saved.name.split("__")[0]
                    counts[lab] = max(0, counts.get(lab, 1) - 1)
                    message, message_until = f"Deleted last take ({lab})", now + 2
                    last_saved = None
            elif state == "idle":
                if key in (9, ord("]")):
                    label_idx = (label_idx + 1) % len(ALL_LABELS)
                elif key == ord("["):
                    label_idx = (label_idx - 1) % len(ALL_LABELS)
                elif key == ord("-"):
                    label_idx = ALL_LABELS.index(IDLE)
                elif 32 < key < 127 and chr(key).upper() in ALL_LABELS:
                    label_idx = ALL_LABELS.index(chr(key).upper())
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    print("\nTakes per label (target in brackets):")
    for lab in ALL_LABELS:
        have = counts.get(lab, 0)
        flag = "" if have >= target_for(lab) else "   <-- need more"
        print(f"  {lab:10s} {have:3d} [{target_for(lab)}]{flag}")


if __name__ == "__main__":
    main()
