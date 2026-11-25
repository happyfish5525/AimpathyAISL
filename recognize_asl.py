"""Real-time ASL recognition: signs from the webcam become editable text.

    python recognize_asl.py                          old window classifier
    python recognize_asl.py --ctc asl_ctc_v7.pth     CTC sequence model reads the letters (bonus 1)

With --ctc, the ABC mode is read by the sequence model: it outputs the letters
(and SPACE) in order from the last 4 s of video. The 123 and WORDS modes still use the
old classifier. In ABC mode the both-fists DELETE sign is not used; press BACKSPACE.

Keys
----
    1 / 2 / 3      jump straight to mode: 1 = ABC (letters), 2 = 123 (digits), 3 = WORDS
    4 or P         PAUSE: ignore all hands (P again, or 1/2/3, resumes)
    TAB            cycle through the modes
    Signs          both open palms up = SPACE,  both fists up = DELETE  (work in every mode)
    SPACE          insert a space            BACKSPACE / DELETE   erase
    ENTER          new line                  LEFT / RIGHT / UP / DOWN   move cursor
    HOME / END     start / end of line       C   clear all
    S              save the text to output.txt
    ESC or Q       quit
    (if your arrow keys do nothing, use  ,  and  .  for left / right)
"""
import argparse
import time
import numpy as np
from collections import deque

import cv2

from model import load_checkpoint, predict_probs
from recognizer import Registrar, SignDecider
from text_editor import TextEditor
from ctc_model import load_checkpoint as load_ctc, predict_paths
from seq_data import T as CTC_T, VOCAB as CTC_VOCAB, WINDOW_SEC as CTC_WINDOW_SEC
from stream_decoder import StreamingDecoder
from utils import build_features, resample_window
from utils import (DYNAMIC_LABELS, DYNAMIC_WINDOW_SEC, MODES, HandTracker,
                   draw_hands, frame_features, open_camera)

PREDICT_EVERY = 0.08          # seconds between network runs
WRAP_CHARS = 34               # characters per line in the text panel
PANEL_LINES = 4
LINE_H = 30

# waitKeyEx codes differ per platform: Windows, then Linux (GTK/Qt).
KEYS = {
    "left": {2424832, 65361}, "up": {2490368, 65362},
    "right": {2555904, 65363}, "down": {2621440, 65364},
    "delete": {3014656, 65535}, "home": {2359296, 65360}, "end": {2293760, 65367},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="asl_model.pth")
    p.add_argument("--ctc", default="", help="CTC checkpoint, e.g. asl_ctc_v7.pth (reads ABC mode)")
    p.add_argument("--landmarker", default="hand_landmarker.task")
    p.add_argument("--camera", type=int, default=0)
    return p.parse_args()


def draw_text_panel(frame, editor, view_top):
    """Bottom text box with wrapping, a visible cursor and auto-scroll."""
    h, w = frame.shape[:2]
    top = h - PANEL_LINES * LINE_H - 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (w, h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    lines = editor.layout(WRAP_CHARS)
    cur_line, cur_col = editor.cursor_line_col(WRAP_CHARS)
    view_top = min(view_top, cur_line)                       # keep cursor visible
    view_top = max(view_top, cur_line - PANEL_LINES + 1)
    view_top = max(0, min(view_top, max(0, len(lines) - PANEL_LINES)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for row, i in enumerate(range(view_top, min(len(lines), view_top + PANEL_LINES))):
        s, e = lines[i]
        text = editor.text[s:e]
        y = top + 24 + row * LINE_H
        cv2.putText(frame, text, (12, y), font, 0.7, (255, 255, 255), 2)
        if i == cur_line and int(time.time() * 2) % 2 == 0:   # blinking cursor
            x = 12 + cv2.getTextSize(text[:cur_col], font, 0.7, 2)[0][0]
            cv2.line(frame, (x + 1, y - 20), (x + 1, y + 5), (0, 255, 255), 2)
    if len(lines) > PANEL_LINES:
        cv2.putText(frame, f"line {cur_line + 1}/{len(lines)}", (w - 120, top + 18),
                    font, 0.45, (0, 255, 255), 1)
    return view_top


def main():
    args = parse_args()
    print(__doc__)
    model, classes = load_checkpoint(args.model)
    decider = SignDecider(lambda f: predict_probs(model, f), classes)
    print(f"Model loaded. Classes: {classes}")
    ctc_model = load_ctc(args.ctc)[0] if args.ctc else None
    if ctc_model is not None:
        print(f"CTC model {args.ctc} reads the ABC mode.")
    ctc_dec, last_ctc, ctc_tentative = StreamingDecoder(), 0.0, ""
    CTC_EVERY = 0.2                       # seconds between CTC runs

    registrar = Registrar()
    editor = TextEditor()
    mode_idx, view_top = 0, 0
    resume_idx, prev_mode_idx = 0, 0      # mode to return to after PAUSE
    buffer = deque()                      # (time, frame vector)
    last_pred = 0.0
    info = {"static": ("-", 0.0), "dynamic": ("-", 0.0), "motion": 0.0,
            "label": None, "conf": 0.0}
    flash_text, flash_until = "", 0.0

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
            draw_hands(frame, result)
            buffer.append((now, frame_features(result)))
            while buffer and now - buffer[0][0] > max(DYNAMIC_WINDOW_SEC, CTC_WINDOW_SEC if ctc_model else 0.0) + 0.5:
                buffer.popleft()

            if MODES[mode_idx] != "PAUSE":
                resume_idx = mode_idx
            if mode_idx != prev_mode_idx:  # fresh start after any mode change
                prev_mode_idx = mode_idx
                buffer.clear()
                registrar = Registrar()
                ctc_dec.reset()
                ctc_tentative = ""
            paused = MODES[mode_idx] == "PAUSE"

            use_ctc = ctc_model is not None and MODES[mode_idx] == "ABC"
            if use_ctc:
                if now - last_ctc >= CTC_EVERY and len(buffer) > 5:
                    last_ctc = now
                    times = np.array([t for t, _ in buffer], dtype=np.float64)
                    feats = np.array([f for _, f in buffer], dtype=np.float32)
                    window = build_features(
                        resample_window(times, feats, now, CTC_WINDOW_SEC, CTC_T))
                    path = predict_paths(ctc_model, window[None])[0]
                    for tok in ctc_dec.update(path, now):
                        ch = CTC_VOCAB[tok]
                        editor.insert(" " if ch == "SPACE" else ch)
                        flash_text, flash_until = ch if ch != "SPACE" else "_", now + 0.6
                    ctc_tentative = ctc_dec.tentative_text
            elif now - last_pred >= PREDICT_EVERY:
                last_pred = now
                times = [t for t, _ in buffer]
                feats = [f for _, f in buffer]
                info = decider.decide(times, feats, now, MODES[mode_idx])
                label = registrar.update(info["label"], now)
                if label is not None:
                    if label == "SPACE":
                        editor.insert(" ")
                    elif label == "DELETE":
                        editor.backspace()
                    elif len(label) == 1:
                        editor.insert(label)
                    else:
                        editor.add_word(label)
                    flash_text, flash_until = label.replace("_", " "), now + 1.0
                    if label in DYNAMIC_LABELS:
                        buffer.clear()   # the sign's tail must not trigger a second sign

            # ---- display ------------------------------------------------
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 62), (25, 25, 25), -1)
            cv2.rectangle(frame, (0, 0), (int(w * registrar.progress), 5), (0, 255, 255), -1)
            s_lab, s_conf = info["static"]
            d_lab, d_conf = info["dynamic"]
            cv2.putText(frame, f"MODE: {MODES[mode_idx]}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if paused else (0, 255, 255), 2)
            cv2.putText(frame, "[1]ABC [2]123 [3]WORDS [4]PAUSE", (w - 290, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            if paused:
                cv2.putText(frame, "PAUSED - hands ignored", (w // 2 - 170, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f"still {s_lab} {s_conf:.2f} | moving {d_lab} {d_conf:.2f} | "
                               f"motion {info['motion']:.3f}", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            if now < flash_until:
                cv2.putText(frame, flash_text, (w // 2 - 80, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
            if use_ctc:
                cv2.putText(frame, f"CTC reading: {ctc_tentative}", (10, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            view_top = draw_text_panel(frame, editor, view_top)
            cv2.imshow("ASL Real-Time Recognition", frame)

            # ---- keyboard -----------------------------------------------
            raw = cv2.waitKeyEx(1)
            if raw == -1:
                continue
            if raw in KEYS["left"]:
                editor.left()
            elif raw in KEYS["right"]:
                editor.right()
            elif raw in KEYS["up"]:
                editor.up(WRAP_CHARS)
            elif raw in KEYS["down"]:
                editor.down(WRAP_CHARS)
            elif raw in KEYS["delete"]:
                editor.delete()
            elif raw in KEYS["home"]:
                editor.home(WRAP_CHARS)
            elif raw in KEYS["end"]:
                editor.end(WRAP_CHARS)
            elif raw < 256:
                ch = raw
                if ch in (27, ord("q")):
                    break
                elif ch == 9:
                    mode_idx = (mode_idx + 1) % len(MODES)
                elif ch in (ord("1"), ord("2"), ord("3"), ord("4")):
                    mode_idx = ch - ord("1")      # 1 ABC, 2 123, 3 WORDS, 4 PAUSE
                elif ch in (ord("p"), ord("P")):
                    mode_idx = resume_idx if paused else MODES.index("PAUSE")
                elif ch == 32:
                    editor.insert(" ")
                elif ch == 13:
                    editor.insert("\n")
                elif ch == 8:
                    editor.backspace()
                elif ch == ord(","):
                    editor.left()
                elif ch == ord("."):
                    editor.right()
                elif ch in (ord("c"), ord("C")):
                    editor.clear()
                elif ch in (ord("s"), ord("S")):
                    with open("output.txt", "w", encoding="utf-8") as f:
                        f.write(editor.text)
                    flash_text, flash_until = "SAVED", now + 1.0
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()