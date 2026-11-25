# ASL Sign Recognition (letters, digits and words → editable text)

Webcam → MediaPipe hand landmarks → time-windowed features → our own
BiGRU network → typed text with cursor, space, delete and line wrapping.

Recognises A–Z (including the moving letters J and Z), 0–9 and the moving signs
HELLO, GOODBYE, THANK YOU, YES, NO, ME, YOU, NAME and HOW ARE YOU, plus two
control signs (SPACE and DELETE). Works with either hand and with one or two hands.

The recognition model (`SignNet` in `model.py`) is designed and trained from scratch.
MediaPipe is used only for hand tracking; no external recognition API is used.

## Quick start (run the trained model)

A trained model (`asl_model.pth`) and the MediaPipe hand model
(`hand_landmarker.task`) are included in this repo, so you can run it right away.

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python recognize_asl.py
```

Keys: **1** letters, **2** digits, **3** words, **4** pause (TAB cycles modes),
**S** saves the text to `output.txt`, **ESC** quits. Full key list is at the top of
`recognize_asl.py`.

If `hand_landmarker.task` is ever missing, download it and put it next to the scripts:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

## How it works

1. **Hand tracking:** MediaPipe returns 21 landmarks per hand for every camera frame.
2. **Features:** landmarks for up to two hands, plus wrist velocity, become one
   136-value vector per frame (`utils.py`). Frames are cut into windows of 30 steps:
   a 1 second window for still signs and a 2 second window for moving signs.
3. **Model:** `BatchNorm → 1D-conv → bidirectional GRU → mean+max pooling → MLP`
   (`model.py`). It is small enough to run on a laptop CPU.
4. **Decision rules:** `recognizer.py` turns the predictions into typed characters.
   A sign must be predicted several times in a row and above a confidence
   threshold before it is typed. Moving signs must be released before they can
   fire again.
5. **Text:** `text_editor.py` handles the cursor, insert, delete and word wrap.
   Both open palms up types a space, and both fists up deletes a character.

Look-alike pairs (0/O, 1/D, 2/V, 6/W, 9/F) are separated by the mode toggle
(letters vs digits), so training scores each pair as one class.

## Results

Accuracy is measured on windows cut from recorded test takes. Look-alike pairs
count as one class. Reproduce with `python evaluate_by_person.py`.

| Test set | Person | Takes / Windows | Accuracy | Note |
|---|---|---|---|---|
| Held-out | `leyu_test` | 244 / 976 | **91.0%** | Never used for training |
| Seen in training | `lls_test` | 917 / 3,668 | **99.2%** | These takes are also in `data/takes`, so this number is optimistic |

* **Held-out (`leyu_test`) weakest classes:** `YOU` (0%, often confused with `Z`), `M` (45%), `X` (60%), `Z` (60%), `GOODBYE` (65%), `NO` (75%), `H` (80%), `IDLE` (80%).
* **Seen in training (`lls_test`) weakest classes:** `T` (90%), `E` (93%), `M` (95%), `N` (95%), `HELLO` (97%), `IDLE` (98%), `7` (98%).

*Last updated: November 20, 2026 (retrain and update these numbers whenever the data changes).*

## Training your own model

### Step 1 – Check the camera and hand tracking
```bash
python webcam_test.py
```
The skeleton should follow your hand. Press **q** to quit.

### Step 2 – Record data
```bash
python collect_data.py --person yourname
```
Pick a label (A–Z, 0–9, TAB for words, `-` for IDLE), press **SPACE**: it records
5 takes of 3 seconds each. Between takes, change something: distance, angle,
height, hand, lighting.

* Still signs: hold the shape (natural wobble is good). Target **10 takes** each.
* Moving signs (J, Z, words): wait for **SIGN NOW!** and perform the sign once. Target **20 takes** each.
* **IDLE** (20+ takes): hand resting, hand moving into position, random fidgeting.
  Needed so the model can answer "nothing" instead of guessing a letter.
* Record some takes with your other hand, and ask friends to record too.
* BACKSPACE deletes the last take. Quit with ESC to see which labels need more.

Takes are saved as `data/takes/LABEL__person__timestamp.npz`.

### Step 3 – Record a separate test set
```bash
python collect_data.py --out data/test --person yourname_test
```
A few takes per label is enough. Keep these takes out of `data/takes`; otherwise
the test score is not honest.

### Step 4 – Train
```bash
python train_model.py
```
Prints validation accuracy (held-out takes), the weakest classes and the most
common mistakes, and saves the best model to `asl_model.pth`. Record more takes
for whatever it confuses, then re-run.

### Step 5 – Evaluate per person
```bash
python evaluate_by_person.py
```
Scores the model separately for each person found in `data/test`.

### Step 6 – Run
```bash
python recognize_asl.py
```

## Adding more people (merging data)

Each take is a standalone `.npz` file (label, timestamps, hand features; no video).
To combine recordings from several people:

1. Everyone uses the same version of this code (the features are computed at recording time).
2. Each person runs `python collect_data.py --person theirname` and zips their `data/takes`.
3. Copy their files into your `data/takes` and run `python train_model.py`.

Label names must match exactly (e.g. `THANK_YOU`, `HOW_ARE_YOU`). Keep at least one
person's takes only in `data/test` so there is always an honest score.

## Tuning without retraining (`recognizer.py`)
| Problem | Change |
|---|---|
| Letters appear too slowly | lower `STABLE_STATIC` |
| Wrong letters sneak in | raise `STATIC_CONF` or `STABLE_STATIC` |
| Letters ignored while hand is still | raise `MOTION_GATE` (watch the "motion" number on screen) |
| Words not detected | lower `DYNAMIC_CONF` |
| Letter doubles unexpectedly | raise `REPEAT_SEC` |

## Adding more words later
1. In `utils.py`, add the word to the `WORDS` list (use `_` for spaces, e.g. `"HOW_ARE_YOU"`).
2. Record it: `python collect_data.py`, press `]` / TAB until the label shows,
   then SPACE. Wait for **SIGN NOW!** and sign once. About 20 takes, varied distance/angle/hand.
3. Retrain: `python train_model.py`. Old takes are kept, nothing needs re-recording.
4. Run `python recognize_asl.py`, press 3 for WORDS.

Choose signs with clearly different movements, and avoid ones that are only a still handshape.

## Control signs: SPACE and DELETE (no keyboard needed)
Both use **two hands**, so they can never be confused with a letter. They work in every mode except PAUSE.

| Sign | Gesture | Behaviour |
|---|---|---|
| SPACE | both open palms up, side by side | one space per gesture; lower your hands to use it again |
| DELETE | both fists up, side by side | deletes one character, repeats while you keep holding |

Record about 10 takes of each with `collect_data.py` (press `]` until the label shows),
then retrain.

## Teaching the model what is NOT a sign (IDLE takes)
False triggers (a phone read as THANK YOU, an I changing to B read as J) are fixed by
recording those exact situations under the **IDLE** label:
* **Transitions:** show an I, switch quickly to another letter, then lower your hand
  out of the frame straight away. Do not hold the new letter. About 25 takes.
* **Everyday movement:** hold a phone up, drink, scratch your face, fix your hair, wave
  something, move both hands freely. About 25 takes.

Retrain afterwards (`python train_model.py`).

## Files
| File | Purpose |
|---|---|
| `utils.py` | labels, feature extraction, windowing, augmentation, camera/MediaPipe helpers |
| `model.py` | network (1D-conv + BiGRU) and checkpoint loading |
| `collect_data.py` | records takes |
| `train_model.py` | trains and evaluates |
| `evaluate_by_person.py` | per-person accuracy on the test folder |
| `recognizer.py` | turns predictions into confirmed signs |
| `text_editor.py` | cursor, editing and word wrap |
| `recognize_asl.py` | live app |
| `webcam_test.py` | camera and hand-tracking check |
| `asl_model.pth` | trained model used by default |
| `hand_landmarker.task` | MediaPipe hand tracking model |
<<<<<<< HEAD
| `data/takes`, `data/test` | recorded training and test takes |
=======
| `data/takes`, `data/test` | recorded training and test takes |
>>>>>>> 1f772e4 (Initial commit: ASL sign language model code and setup)
