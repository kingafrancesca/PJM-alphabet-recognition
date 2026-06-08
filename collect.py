"""zbieranie wlasnych probek landmarkow - personalizacja modelu pod konkretna dlon.
model z Kaggle myli litery na zywo (domain shift), bo nie zna tej dloni.
ten skrypt nagrywa Twoje pozy do data/personal_samples.csv w tym samym formacie co
oryginalny CSV (kolumny 0_point_lm_*), zeby training.py mogl je dolaczyc.

obsluga:
  litera a-z   - wybor nagrywanej litery (pokazana u gory)
  SPACJA       - start/stop nagrywania (trzymaj poze, klatki leca do bufora)
  BACKSPACE    - usun ostatnia probke wybranej litery
  s            - zapis do CSV
  q            - zapis i wyjscie
celuj w ~60-100 probek na litere, z lekko roznych katow/odlegosci."""

import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_MODEL = Path(__file__).parent / "hand_landmarker.task"
OUT_CSV = Path(__file__).parent / "data" / "personal_samples.csv"

# dokladnie te kolumny czyta training.py - 21 landmarkow x/y/z
LM_COLUMNS = [f"0_point_lm_{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]

# litery bazowe ktore zna model (j/ch/cz/rz sa dynamiczne i odrzucane w treningu)
VALID = set("abcdefghiklmnoprstuwyz")

# dwuznaki o wlasnym, charakterystycznym ksztalcie - maja wlasna klase (nie baza+ruch).
# wybierane klawiszem cyfrowym, bo etykieta jest wieloznakowa
EXTRA_LABELS = {"1": "sz"}

# diakrytyki/dwuznaki bez wlasnej klasy = baza + ruch (decyzja w rules.py).
# nagrywaj poze pod litera bazowa o tym samym KSZTALCIE dloni:
DIACRITIC_HINT = "ą=a ę=e ć=c ł=l ń=n ó=o ś=s  ch/cz=h  j=i  ż/ź=z   sz=klawisz 1"


def load_existing():
    """wczytuje dotychczasowe probki - dopisujemy, nie nadpisujemy"""
    rows = []
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                rows.append(row)
    return rows


def counts(rows):
    out = {}
    for row in rows:
        out[row["label"]] = out.get(row["label"], 0) + 1
    return out


def save(rows):
    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label"] + LM_COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[zapis] {len(rows)} probek -> {OUT_CSV.name}")


def row_from_landmarks(label, landmarks):
    row = {"label": label}
    for i, lm in enumerate(landmarks):
        row[f"0_point_lm_{i}_x"] = lm.x
        row[f"0_point_lm_{i}_y"] = lm.y
        row[f"0_point_lm_{i}_z"] = lm.z
    return row


def main():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    rows = load_existing()
    print(f"wczytano {len(rows)} istniejacych probek")

    cap = cv2.VideoCapture(0)
    label = None
    recording = False
    frame_num = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(image, frame_num * 33)

        has_hand = bool(result.hand_landmarks)
        if recording and has_hand and label is not None:
            rows.append(row_from_landmarks(label, result.hand_landmarks[0]))

        # HUD
        c = counts(rows)
        target = f"litera: {label.upper()}  ({c.get(label, 0)} probek)" if label else "litera: (wybierz a-z)"
        cv2.putText(frame, target, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2)
        status = "NAGRYWAM" if recording else "pauza (SPACJA=start)"
        color = (0, 0, 255) if recording else (200, 200, 200)
        cv2.putText(frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if not has_hand:
            cv2.putText(frame, "brak dloni", (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 165, 255), 2)
        cv2.putText(frame, f"razem: {len(rows)}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, DIACRITIC_HINT, (20, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("zbieranie probek PJM", frame)
        key = cv2.waitKey(1) & 0xFF
        frame_num += 1

        if key == ord("q"):
            break
        if key == ord("s"):
            save(rows)
        if key == ord(" "):
            recording = not recording
        if key == 8 and label is not None:  # BACKSPACE - usun ostatnia probke tej litery
            for i in range(len(rows) - 1, -1, -1):
                if rows[i]["label"] == label:
                    del rows[i]
                    break
        ch = chr(key) if 32 <= key < 127 else ""
        if ch in VALID:
            label = ch
            recording = False  # zmiana litery wstrzymuje nagrywanie
        elif ch in EXTRA_LABELS:
            label = EXTRA_LABELS[ch]  # dwuznak z wlasna klasa, np. sz
            recording = False

        # zamkniecie krzyzykiem
        if cv2.getWindowProperty("zbieranie probek PJM", cv2.WND_PROP_VISIBLE) < 1:
            break

    save(rows)
    cap.release()
    cv2.destroyAllWindows()
    print("\nrozklad zebranych probek:")
    for letter in sorted(counts(rows)):
        print(f"  {letter}: {counts(rows)[letter]}")


if __name__ == "__main__":
    main()
