# Aimpathy Bridge — Real-Time ASL Fingerspelling Recognition

Real-time American Sign Language (ASL) fingerspelling and sign recognition built with **MediaPipe Hand Landmarker** (`136` engineered features per frame) and a **PyTorch Conv1D + BiGRU + CTC** sequence model (`SeqNet`) with hands-free live text editing.

## Repository Structure

```text
Aimpathy_AI_SignLanguage_Model/
├── notebooks/               # Jupyter notebooks (dataset prep, baseline, CTC eval, inference, history)
├── scripts/                 # CLI entrypoints (data collection, training, evaluation, live webcam demo)
├── src/                     # Core model architectures, feature extraction, and streaming CTC decoder
├── trained_models/
│   └── ctc/
│       └── asl_ctc_v6.pth   # Stable production CTC checkpoint
├── hand_landmarker.task     # MediaPipe hand landmark model
├── requirements.txt
└── README.md
```

## Quick Start (Live Webcam Demo)

From the repository root in Windows Command Prompt (`cmd`):

```cmd
python scripts\\recognize_asl.py --ctc "trained_models\\ctc\\asl_ctc_v6.pth"
```

## Benchmark Results (`data/words_test` — 91 Real Word Takes, 30 Unique Words)

Evaluated sequentially in chronological order without future lookahead (`scripts/eval_words.py`):

| Pipeline | Model | Character Error Rate (CER) | Exact-Word Accuracy |
| :--- | :--- | :---: | :---: |
| **OLD (Baseline)** | `SignNet` window classifier + rules (`ABC` mode) | 73.8% | 1.1% |
| **CTC (Current `v6`)** | `SeqNet` (`Conv1D + BiGRU`) + `StreamingDecoder` | **34.5%** | **15.4%** |

## Key Commands

* **Evaluate CTC model on real recorded words:**
  ```cmd
  python scripts\\eval_words.py --words-dir "data\\words_test" --ctc "trained_models\\ctc\\asl_ctc_v6.pth" --verbose
  ```
* **Train / fine-tune CTC sequence model (saving to a new checkpoint):**
  ```cmd
  python scripts\\train_ctc.py --init "trained_models\\ctc\\asl_ctc_v6.pth" --out "trained_models\\ctc\\asl_ctc_v7.pth" --words-dir "data\\words_train" --exclude-words-dir "data\\words_test" --lr 0.0003 --epochs 20
  ```
