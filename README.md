# PJM Alphabet Recognition

Real-time Polish Sign Language (PJM) fingerspelling recognition from a webcam.
Combines MediaPipe hand landmarks with a Random Forest classifier and a
motion-based heuristic to handle diacritics and digraphs, with text-to-speech output.

## What it does

- Detects a hand in the webcam feed and recognizes the PJM letter being shown
- Handles diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż) and digraphs (ch, cz, rz, sz, j)
  by analysing hand motion on top of the static pose
- Draws the recognized letter on screen and reads it out loud
- Stabilizes predictions across frames so the output does not flicker

## How it works

The pipeline is split into three stages:

**Syntax** — MediaPipe `HandLandmarker` extracts 21 hand landmarks from each frame.

**Semantics** — the landmarks are normalized relative to the wrist and hand size,
then a Random Forest classifies them into one of 22 base letters. Normalizing the
features makes recognition independent of where the hand is in the frame and how
far it is from the camera.

**Pragmatics** — a motion heuristic looks at how the hand moved over the last few
frames. The base letter plus the motion description decides the final character
(for example `a` plus a downward motion becomes `ą`). The result is shown on screen
and sent to a text-to-speech engine running on a separate thread.

## Requirements

- **Python 3.10** — MediaPipe does not support newer versions
- Windows — the text-to-speech and font handling are Windows-specific
- Packages: `opencv-python`, `mediapipe`, `scikit-learn`, `pandas`, `numpy`,
  `joblib`, `pyttsx3`, `Pillow`, `pywin32`

## Setup

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install opencv-python mediapipe scikit-learn pandas numpy joblib pyttsx3 Pillow pywin32
```

## Usage

```powershell
# one-time: download the MediaPipe hand landmarker model
python download_model.py

# run live recognition from the webcam
python camera.py
```

Controls in the camera window:

- `q` — quit
- `f` — toggle fullscreen
- `d` — print a snapshot (top-5 letter probabilities + motion numbers) to the console,
  used for calibrating the motion thresholds

### Personalizing the model to your own hand

The base Kaggle model suffers from domain shift — it works on the dataset but confuses
letters on a hand it has never seen. To fix this, record your own samples and retrain:

```powershell
python collect.py     # record landmarks per letter (a-z select, SPACE rec, 1 = sz, s save, q quit)
python training.py    # auto-includes data/personal_samples.csv and rebuilds the model
```

This raised accuracy from 91.7% to 97.3% and removed almost all live misclassifications.

The trained `model.joblib` is not checked in (it is git-ignored). To produce it,
download the dataset (see below), put the CSV in `data/`, and run:

```powershell
python training.py
```

## Model

Random Forest (300 trees) on 126 normalized features per frame.

- **Accuracy: 97.3%** on a held-out 20% test split (2856 samples, 23 classes), after
  personalization. The base Kaggle-only model scored 91.7% but failed live on an unseen
  hand; adding the user's own samples (`collect.py`) fixed the domain shift.
- Previously weak letters recovered: `t` 0.62→0.96, `d` 0.75→0.96, `s` 0.60→0.92.
- Hardest remaining pair: `o`↔`s` (objectively similar static poses).

Most diacritics and digraphs are not separate classes: the classifier predicts the base
letter and `rules.py` upgrades it based on the detected motion. The exception is `sz`,
which has a distinctive handshape and gets its own class.

## Calibration

The motion thresholds in `rules.py` are per-letter, because the range of motion
differs between gestures. They were tuned from the `d`-key snapshots and `[debug]`
console logs on real gestures: the resting "noise" level turned out to be ~0.1–0.15
and real gestures 0.3–1.9, so the original 0.16 thresholds were too low and caused
flicker between base letters and diacritics. Some letters are split by motion
**direction** (e.g. `ch` = down, `cz` = sideways — same handshape, base `h`).

`Z`/`Ź`/`Ż` remain the weakest case: they differ by the number of motion reversals,
but drawing a "Z" in the air inherently produces reversals, and the sliding 15-frame
window catches different parts of the gesture. See RAPORT.md for the full discussion
and the LSTM redesign that would solve it.

## Tests

```powershell
pip install pytest
pytest
```

Covers the motion description, the rule engine, the prediction/frame buffers, and the
threaded speech queue (TTS is stubbed, so the tests run silently).

## Dataset

Trained on the [Polish Sign Language landmarks dataset](https://www.kaggle.com/datasets/kacperjarosik1/polish-sign-language-google-landmarks-csv-data) from Kaggle.

Built by Kinga Kinowska.
