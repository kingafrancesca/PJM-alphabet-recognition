import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEST = Path(__file__).parent / "hand_landmarker.task"

print(f"pobieram do {DEST}...")
urllib.request.urlretrieve(URL, DEST)
print(f"gotowe, rozmiar: {DEST.stat().st_size / 1024 / 1024:.2f} MB")
