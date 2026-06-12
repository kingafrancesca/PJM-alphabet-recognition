# PJM Alphabet Recognition

Real-time Polish Sign Language (PJM) fingerspelling recognition from a webcam.
MediaPipe finds the hand, a Random Forest classifies the base letter and a small
motion heuristic turns it into the final character (diacritics and digraphs).
The result is shown on screen and read out loud.

## What it does

- detects a hand in the webcam feed and recognizes the PJM letter being shown
- handles diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż) and digraphs (ch, cz, rz, sz, j)
  by looking at hand motion on top of the static pose
- draws the recognized letter on screen and reads it out loud
- stabilizes predictions across frames so the output does not flicker

## How it works

MediaPipe HandLandmarker extracts 21 hand landmarks from each frame. The landmarks
are normalized relative to the wrist and hand size, so recognition does not depend
on where the hand is in the frame or how far it is from the camera. A Random Forest
classifies the normalized features into one of the base letters. Then a motion
heuristic looks at how the hand moved over the last frames and picks the final
character, for example "a" plus a downward motion becomes "ą". The recognized
character goes to the screen and to a text-to-speech engine running on a separate
thread.

## Requirements

- Python 3.10 (MediaPipe does not support newer versions)
- Windows (the text-to-speech and font handling are Windows-specific)
- packages: `opencv-python`, `mediapipe`, `scikit-learn`, `pandas`, `numpy`,
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

- `q` - quit
- `f` - toggle fullscreen
- `i` - show/hide the info overlays (confidence ranking + motion description)
- `d` - print a snapshot (top-5 letter probabilities + motion numbers) to the
  console, useful when calibrating the motion thresholds

### Personalizing the model to your own hand

The base Kaggle model suffers from domain shift: it scores well on its own test
split but confuses letters on a hand it has never seen. The fix is to record your
own samples and retrain:

```powershell
python collect.py     # record landmarks per letter (a-z select, SPACE rec, 1 = sz, s save, q quit)
python training.py    # auto-includes data/personal_samples.csv and rebuilds the model
```

In my case that meant about 14 thousand extra samples. Accuracy went from 92% to
98.2% and almost all live misclassifications disappeared.

The trained `model.joblib` is not checked in (it is git-ignored). To produce it,
download the dataset (see below), put the CSV in `data/`, and run `python training.py`.

## Model

Random Forest (300 trees) on 126 normalized features per frame.

Accuracy after personalization: 98.2% on a held-out 20% test split (3517 samples,
23 classes). The Kaggle-only model scored 92% on its own split but failed live on
an unseen hand. The letters that recovered the most (F1 before and after):
s 0.54 to 0.96, t 0.67 to 0.98, d 0.76 to 0.97.

Since most of the training data comes from a single hand, 98.2% reflects accuracy
for the calibrated user. For anyone else, 92% is the more honest estimate.

Most diacritics and digraphs are not separate classes: the classifier predicts the
base letter and `rules.py` upgrades it based on the detected motion. The exception
is `sz`, which has a distinctive handshape and gets its own class.

## Calibration

The motion thresholds in `rules.py` are per-letter, because the range of motion
differs between gestures. They were tuned from the `d`-key snapshots and `[debug]`
console logs on real gestures: the resting "noise" level turned out to be around
0.1-0.15 and real gestures 0.3-1.9, so the original 0.16 thresholds were too low
and caused flicker between base letters and diacritics. Some letters are split by
motion direction (e.g. `ch` = down, `cz` = sideways, same handshape, base `h`).

`Z`/`Ź`/`Ż` remain the weakest case. They differ by the number of motion reversals,
but drawing a "Z" in the air produces reversals by itself, and the sliding frame
window catches different parts of the gesture. A sequence model (LSTM) trained on
full motion recordings would solve it, but that is out of scope for this version.

## Tests

```powershell
pip install pytest
pytest
```

Covers the motion description, the rule engine, the prediction/frame buffers, and
the threaded speech queue.

## Dataset

Trained on the [Polish Sign Language landmarks dataset](https://www.kaggle.com/datasets/kacperjarosik1/polish-sign-language-google-landmarks-csv-data) from Kaggle and personally collected samples.

Built by Kinga Kinowska.
