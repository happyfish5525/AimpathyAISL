from pathlib import Path
import sys

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

"""Record REAL fingerspelled words (several letters in a row, no pauses).

    python collect_words.py --person yourname --out data/words_train
    python collect_words.py --person friend_test --out data/words_test

Why: the CTC model is trained on words stitched from single-letter takes.
Real spelling has things the stitching only imitates (how fast letters
change, what a double letter looks like). Real words are therefore
  * the HONEST test set (record them with a signer the model never trained on)
  * extra training data for fine-tuning (record them with a training signer).

Each take is saved like collect_data.py saves its takes, with the whole word
as the label:  data/words_train/HELLO__yourname__1795230000000.npz

How to spell: sign the letters one after another at your normal speed. Do not
hold each letter for a long time and do not stop between letters. For a double
letter (LL) just sign the letter twice, the way you naturally would.

Keys
----
    SPACE        record the shown word      N or TAB   next word
    P            previous word              BACKSPACE  delete the last saved take
    ESC          quit
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from utils import PRESENCE_COLS, HandTracker, draw_hands, frame_features, open_camera

# Words with double letters and with the hard letters (J, Z, M, N, X, Y) first.
DEFAULT_WORDS = [
    "HELLO", "COFFEE", "BOOK", "SEE", "ALL", "LITTLE", "SUMMER", "HAPPY",
    "APPLE", "SCHOOL", "JAZZ", "ZEBRA", "JUMP", "QUICK", "MOON", "NAME",
    "MILK", "WATER", "FRIEND", "FAMILY", "HOUSE", "GREEN", "SORRY", "PLEASE",
    "YELLOW", "EXAM", "TEA", "RICE", "WORK", "LOVE",
]
GET_READY_SEC = 1.5
LEAD_SEC = 1.0            # time to start spelling after the recording begins
SEC_PER_LETTER = 0.8      # recording length = LEAD_SEC + this * number of letters
MIN_VISIBLE = 0.60        # discard takes where the hand was visible less than this
TARGET_TAKES = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/words_train")
    p.add_argument("--person", default="me")
    p.add_argument("--words", default="", help="comma separated list to use instead of the default")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--landmarker", default="hand_landmarker.task")
    return p.parse_args()


def count_takes(folder: Path) -> dict:
    counts = {}
    for f in folder.glob("*.npz"):
        word = f.name.split("__")[0]
        counts[word] = counts.get(word, 0) + 1
    return counts


def save_take(folder: Path, word: str, person: str, times, feats):
    feats = np.stack(feats).astype(np.float32)
    times = np.asarray(times, dtype=np.float32)
    visible = float(feats[:, PRESENCE_COLS].any(axis=1).mean())
    fps = len(times) / max(float(times[-1]), 1e-6)
    if visible < MIN_VISIBLE:
        return None, f"Discarded: hand visible only {visible:.0%} of the time"
    path = folder / f"{word}__{person}__{int(time.time() * 1000)}.npz"
    np.savez_compressed(path, times=times, feats=feats, label=word, person=person)
    warn = "  (LOW FPS - close other programs)" if fps < 10 else ""
    return path, f"Saved ({visible:.0%} visible, {fps:.0f} fps){warn}"


def put(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    args = parse_args()
    words = [w.strip().upper() for w in args.words.split(",") if w.strip()] or DEFAULT_WORDS
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    counts = count_takes(folder)

    idx, state, phase_start = 0, "idle", 0.0       # idle -> wait -> record -> idle
    times, feats, last_saved = [], [], None
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
            word = words[idx]
            rec_sec = LEAD_SEC + SEC_PER_LETTER * len(word)

            if state == "wait" and now - phase_start >= GET_READY_SEC:
                state, phase_start, times, feats = "record", now, [], []
            elif state == "record":
                times.append(now - phase_start)
                feats.append(vec)
                if now - phase_start >= rec_sec:
                    path, message = save_take(folder, word, args.person, times, feats)
                    message_until = now + 3
                    if path is not None:
                        last_saved = path
                        counts[word] = counts.get(word, 0) + 1
                    state = "idle"

            n = counts.get(word, 0)
            put(frame, f"WORD: {word}", 32, (255, 255, 0), 1.0)
            put(frame, f"Takes: {n}/{TARGET_TAKES}   person: {args.person}", 62,
                (0, 255, 0) if n >= TARGET_TAKES else (0, 165, 255))
            if state == "idle":
                put(frame, "SPACE = record   N = next word   P = previous", 90, (200, 200, 200), 0.5)
            elif state == "wait":
                put(frame, "Get ready... hand in view", 90, (0, 255, 255))
            else:
                t = now - phase_start
                put(frame, "SPELL NOW - letters one after another" if t >= 0.3 else "GO", 90,
                    (0, 255, 0), 0.7)
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, h - 14), (int(w * t / rec_sec), h), (0, 0, 255), -1)
            if now < message_until:
                put(frame, message, frame.shape[0] - 40, (0, 255, 255), 0.55)

            cv2.imshow("ASL word recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32 and state == "idle":
                state, phase_start = "wait", now
            elif key == 8:
                if state != "idle":
                    state = "idle"
                    message, message_until = "Cancelled", now + 2
                elif last_saved is not None and last_saved.exists():
                    last_saved.unlink()
                    lab = last_saved.name.split("__")[0]
                    counts[lab] = max(0, counts.get(lab, 1) - 1)
                    message, message_until = f"Deleted last take ({lab})", now + 2
                    last_saved = None
            elif state == "idle":
                if key in (ord("n"), ord("N"), 9):
                    idx = (idx + 1) % len(words)
                elif key in (ord("p"), ord("P")):
                    idx = (idx - 1) % len(words)
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    print(f"\nTakes per word in {folder} (target {TARGET_TAKES}):")
    for w in words:
        have = counts.get(w, 0)
        print(f"  {w:10s} {have:2d}" + ("" if have >= TARGET_TAKES else "   <-- need more"))


if __name__ == "__main__":
    main()
