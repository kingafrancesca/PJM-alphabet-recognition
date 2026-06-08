from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

from buffer import Buffer
from frame_buffer import FrameBuffer
from features import get_features
from speech import Speech
from rules import apply_rules
from motion import describe_motion

HAND_MODEL = Path(__file__).parent / "hand_landmarker.task"
PJM_MODEL = Path(__file__).parent / "model.joblib"

CONFIDENCE_THRESHOLD = 0.25   # min pewnosc pojedynczej predykcji - bufor dominacji i tak filtruje
                              # slabsze litery (o, g, s, t) daja max ~0.30, prog 0.4 je gubil
BUFFER_SIZE = 10              # klatki do stabilizacji predykcji
DOMINANCE_THRESHOLD = 0.7     # ile % bufora musi pokazac ta sama litera
FRAME_BUFFER_SIZE = 15        # klatki landmarkow do wykrywania ruchu - do kalibracji

FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
LETTER_FONT = (ImageFont.truetype(str(FONT_PATH), 90)
               if FONT_PATH.exists() else ImageFont.load_default())

# pary landmarkow do narysowania szkieletu dloni
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def draw_letter(frame, letter):
    """rysuje finalny znak przez PIL - OpenCV nie renderuje polskich liter (ą, ć, ż...)"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text((20, 15), letter.upper(), font=LETTER_FONT,
                               fill=(255, 255, 255))
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_debug(frame, proba, classes):
    """top-3 klasy z prawdopodobienstwami - do kalibracji progow pewnosci"""
    ranking = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    for i, (letter, p) in enumerate(ranking[:3]):
        color = (0, 255, 0) if i == 0 else (180, 180, 180)
        cv2.putText(frame, f"{letter.upper()} {p:.2f}", (20, 150 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def draw_motion(frame, motion):
    """opis ruchu na zywo - do kalibracji progow w motion.py i rules.py"""
    if motion is None:
        text = "ruch: bufor niepelny"
    else:
        text = (f"ruch: {motion['direction']}  ampl {motion['amplitude']:.2f}  "
                f"zakres {motion['range']:.2f}  zawr {motion['reversals']}")
    cv2.putText(frame, text, (20, 275), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)


def print_snapshot(proba, classes, motion):
    """zrzut na klawisz 'd' - komplet liczb do kalibracji progow"""
    print("\n--- snapshot ---")
    if proba is None:
        print("  brak dloni w kadrze")
    else:
        ranking = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
        print("  top5: " + "   ".join(f"{letter}:{p:.2f}" for letter, p in ranking[:5]))
    print(f"  ruch:  {motion}")


def main():
    bundle = joblib.load(PJM_MODEL)
    classifier = bundle["model"]

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    buffer = Buffer(size=BUFFER_SIZE, dominance_threshold=DOMINANCE_THRESHOLD)
    frame_buffer = FrameBuffer(size=FRAME_BUFFER_SIZE)
    speech = Speech(delay=2.0)

    cap = cv2.VideoCapture(0)
    # WINDOW_NORMAL - okno mozna skalowac i przelaczac na pelny ekran (klawisz 'f')
    cv2.namedWindow("PJM", cv2.WINDOW_NORMAL)
    fullscreen = False

    frame_num = 0
    proba = None        # ostatnie prawdopodobienstwa - dla zrzutu na klawisz 'd'
    last_logged = None  # ostatnio zalogowana litera

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(image, frame_num * 33)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks)
            frame_buffer.add(landmarks)

            features = get_features(landmarks)
            proba = classifier.predict_proba([features])[0]
            idx = proba.argmax()
            confidence = proba[idx]
            letter = classifier.classes_[idx]

            draw_debug(frame, proba, classifier.classes_)

            # do bufora tylko jak pewnosc wystarczajaca
            if confidence >= CONFIDENCE_THRESHOLD:
                buffer.add(letter)
            else:
                buffer.add(None)  # niepewna klatka tez sie liczy
        else:
            buffer.clear()          # brak dloni - czyscimy od razu, litera nie wisi
            frame_buffer.add(None)  # przerywamy ciaglosc ruchu
            speech.reset()          # pozwalamy przeczytac te sama litere przy ponownym pokazaniu
            proba = None

        # opis ruchu liczona co klatke - zeby liczby byly widoczne na zywo przy kalibracji
        motion = describe_motion(frame_buffer)
        draw_motion(frame, motion)

        stable = buffer.letter()
        if stable is not None:
            final = apply_rules(stable, motion)
            draw_letter(frame, final)
            speech.say(final)
            # auto-log przy zmianie litery- komplet liczb do kalibracji
            if final != last_logged:
                print(f"[debug] baza={stable} -> {final}   ruch={motion}")
                last_logged = final
        else:
            speech.reset()      # niestabilna predykcja - nie blokujemy ponownego odczytu
            last_logged = None

        cv2.imshow("PJM", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # zamkniecie okna krzyzykiem - getWindowProperty spada ponizej 1 gdy okno znika
        if cv2.getWindowProperty("PJM", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("f"):  # przelacz pelny ekran
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                "PJM", cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        if key == ord("d"):
            print_snapshot(proba, classifier.classes_, motion)
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
