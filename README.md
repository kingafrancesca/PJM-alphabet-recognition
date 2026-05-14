  # PJM Alphabet Recognition

  Real-time Polish Sign Language (PJM) fingerspelling
  recognition from a webcam.
  Combines MediaPipe hand landmarks with a Random Forest
  classifier and a
  motion-based heuristic to handle diacritics and digraphs,
  with text-to-speech output.

  ## What it does

  - Detects a hand in the webcam feed and recognizes the PJM
  letter being shown
  - Handles diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż) and digraphs
   (ch, cz, rz, sz, j)
    by analysing hand motion on top of the static pose
  - Draws the recognized letter on screen and reads it out loud
  - Stabilizes predictions across frames so the output does not
   flicker

  ## How it works

  The pipeline is split into three stages:

  **Syntax** - MediaPipe `HandLandmarker` extracts 21 hand
  landmarks from each frame.

  **Semantics** - the landmarks are normalized relative to the
  wrist and hand size,
  then a Random Forest classifies them into one of 22 base
  letters. Normalizing the
  features makes recognition independent of where the hand is
  in the frame and how
  far it is from the camera.

  **Pragmatics** - a motion heuristic looks at how the hand
  moved over the last few
  frames. The base letter plus the motion description decides
  the final character
  (for example `a` plus a downward motion becomes `ą`). The
  result is shown on screen
  and sent to a text-to-speech engine running on a separate
  thread.

  ## Requirements

  - **Python 3.10** - MediaPipe does not support newer versions
  - Windows - the text-to-speech and font handling are
  Windows-specific
  - Packages: `opencv-python`, `mediapipe`, `scikit-learn`,
  `pandas`, `numpy`,
    `joblib`, `pyttsx3`, `Pillow`, `pywin32`

  ## Setup

  ```powershell
  py -3.10 -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install opencv-python mediapipe scikit-learn pandas numpy
   joblib pyttsx3 Pillow pywin32
  ```

  ## Dataset

  Trained on the [Polish Sign Language landmarks  dataset](https://www.kaggle.com/datasets/kacperjarosik1/polish-sign-language-google-landmarks-csv-data) from Kaggle.

Built by Kinga Kinowska.
