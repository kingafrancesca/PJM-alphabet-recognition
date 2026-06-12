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

CONFIDENCE_THRESHOLD = 0.48 # minimalna pewnosc co do litery, wartosci mniejsze traktuje jako brak litery
BUFFER_SIZE = 10 # klatki do stabilizacji predykcji
DOMINANCE_THRESHOLD = 0.7 # ile % bufora musi pokazac ta sama litera
FRAME_BUFFER_SIZE = 15 # klatki landmarkow do wykrywania ruchu (polskie litery)
Z_FRAME_BUFFER_SIZE = 30 # dluzszy bufor dla ź/ż
COMMIT_FRAMES = 4 # ile klatek finalny znak musi sie utrzymac, zanim go wypowiemy; zapobiega falszywym odczytom w trakcie przejsc.
STATIC_MAX_MOTION = 0.22  # litera statyczna jest rozpoznawana tylko, gdy reka spokojna; blokuje odczyt liter, gdy reka jest w ruchu/przejsciu.

FONTS_DIR = Path("C:/Windows/Fonts")


def _font(name, size, fallback="arial.ttf"):
    for filename in (name, fallback):
        path = FONTS_DIR / filename
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()

LETTER_FONT = _font("segoeuib.ttf", 64, "arialbd.ttf")
LABEL_FONT = _font("segoeui.ttf", 16)
INFO_FONT = _font("segoeui.ttf", 18)
SMALL_FONT = _font("segoeui.ttf", 16)

WIN_WIDTH, WIN_HEIGHT = 1280, 720

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
    for a, b in CONNECTIONS:                          # kolory BGR
        cv2.line(frame, points[a], points[b], (150, 220, 120), 2, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(frame, (x, y), 5, (70, 180, 255), -1, cv2.LINE_AA)


def _panel(draw, box, radius=18, fill=(18, 18, 26, 170)):
    try:
        draw.rounded_rectangle(box, radius=radius, fill=fill)
    except AttributeError:
        draw.rectangle(box, fill=fill)


def render_overlay(frame, letter, ranking, motion, show_info):
    base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = base.size

    # panel z rozpoznana litera
    if letter:
        pad = 16
        px0 = 20
        label = "ROZPOZNANO"
        lbox = draw.textbbox((0, 0), label, font=LABEL_FONT)
        lw = lbox[2] - lbox[0]
        cbox = draw.textbbox((0, 0), letter.upper(), font=LETTER_FONT)
        cw = cbox[2] - cbox[0]
        panel_w = max(lw, cw) + 2 * pad
        _panel(draw, (px0, 20, px0 + panel_w, 20 + 112))
        draw.text((px0 + (panel_w - lw) / 2 - lbox[0], 28), label,
                  font=LABEL_FONT, fill=(140, 195, 255, 255))
        draw.text((px0 + (panel_w - cw) / 2 - cbox[0], 48), letter.upper(),
                  font=LETTER_FONT, fill=(255, 255, 255, 255))

    # ranking top3 najbardziej prawdopodobne litery (ukrywane/odslaniane po wcisnieciu i)
    if show_info and ranking:
        pw = 190
        px = w - 20 - pw
        rows = ranking[:3]
        _panel(draw, (px, 20, px + pw, 20 + len(rows) * 28 + 12))
        for i, (lab, p) in enumerate(rows):
            y = 20 + 12 + i * 28
            color = (120, 230, 150, 255) if i == 0 else (185, 185, 195, 255)
            draw.text((px + 14, y - 3), lab.upper(), font=INFO_FONT, fill=color)
            bar_x = px + 50
            draw.rectangle((bar_x, y + 3, bar_x + 90, y + 13), fill=(55, 55, 66, 220))
            draw.rectangle((bar_x, y + 3, bar_x + int(90 * float(p)), y + 13), fill=color)
            draw.text((bar_x + 100, y - 3), f"{p:.2f}", font=SMALL_FONT, fill=color)

    # opis ruchu
    if show_info:
        if motion is None:
            mtext = "ruch: —"
        else:
            mtext = (f"ruch: {motion['direction']}    ampl {motion['amplitude']:.2f}    "
                     f"zakres {motion['range']:.2f}    zawr {motion['reversals']}")
        mb = draw.textbbox((0, 0), mtext, font=SMALL_FONT)
        _panel(draw, (24, h - 52, 24 + (mb[2] - mb[0]) + 28, h - 52 + 34), radius=14)
        draw.text((38, h - 46), mtext, font=SMALL_FONT, fill=(225, 225, 230, 255))

    # legenda
    hint = "q - wyjscie   f - pelny ekran   i - info   d - snapshot"
    hb = draw.textbbox((0, 0), hint, font=SMALL_FONT)
    draw.text((w - (hb[2] - hb[0]) - 26, h - 40), hint, font=SMALL_FONT,
              fill=(170, 170, 180, 230))

    out = Image.alpha_composite(base, layer).convert("RGB")
    frame[:] = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def print_snapshot(proba, classes, motion):
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
    z_frame_buffer = FrameBuffer(size=Z_FRAME_BUFFER_SIZE)
    speech = Speech()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, "Nie udalo sie otworzyc kamery.",
                                         "PJM - kamera niedostepna", 0x10)
        cap.release()
        return
    cv2.namedWindow("PJM", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cam_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
    cam_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480
    scale = WIN_HEIGHT / cam_h
    cv2.resizeWindow("PJM", int(cam_w * scale), int(cam_h * scale))
    fullscreen = False
    show_info = True

    frame_num = 0
    proba = None
    last_logged = None

    # stan kontroli wypowiadania litery przez syntezator mowy
    candidate = None
    candidate_count = 0
    committed = None
    suppressed_base = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(image, frame_num * 33)

        ranking = None
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks)
            frame_buffer.add(landmarks)
            z_frame_buffer.add(landmarks)

            features = get_features(landmarks)
            proba = classifier.predict_proba([features])[0]
            ranking = sorted(zip(classifier.classes_, proba),
                             key=lambda x: x[1], reverse=True)
            idx = proba.argmax()
            confidence = proba[idx]
            letter = classifier.classes_[idx]

            # do bufora tylko jak pewnosc wystarczajaca
            if confidence >= CONFIDENCE_THRESHOLD:
                buffer.add(letter)
            else:
                buffer.add(None)
        else:
            buffer.clear()          # brak dloni -> czyszczenie buffera
            frame_buffer.add(None)  # przerywamy ciaglosc ruchu
            z_frame_buffer.add(None)
            speech.reset() # zgoda na przeczytanie tej samej litery przy ponownym pokazaniu
            proba = None
            # reka zeszla z kadru -> pelny reset stanu
            candidate = None
            candidate_count = 0
            committed = None
            suppressed_base = None
            last_logged = None

        # opis ruchu liczony co klatke
        motion = describe_motion(frame_buffer)
        z_motion = describe_motion(z_frame_buffer)

        stable = buffer.letter()
        final = apply_rules(stable, motion, z_motion) if stable is not None else None
        shown_motion = z_motion if stable == "z" else motion

        # kontrola wypowiadania liter przez syntezator mowy
        if final == candidate:
            candidate_count += 1
        else:
            candidate = final
            candidate_count = 1

        if final is not None:
            amplitude = motion["amplitude"] if motion else 0.0
            is_gesture = final != stable   # diakrytyk/dwuznak powstaly z ruchu (np. a->ą)
            ready = (
                candidate_count >= COMMIT_FRAMES
                and candidate != committed
                and (is_gesture or amplitude <= STATIC_MAX_MOTION)
            )
            if ready:
                if candidate == suppressed_base:
                    committed = candidate
                else:
                    speech.say(candidate)
                    committed = candidate
                    suppressed_base = stable if is_gesture else None
                if final != last_logged:
                    print(f"[debug] baza={stable} -> {final}  ruch={shown_motion}")
                    last_logged = final

        render_overlay(frame, final, ranking, shown_motion, show_info)
        cv2.imshow("PJM", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty("PJM", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("f"):  # przelacz pelny ekran
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                "PJM", cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        if key == ord("i"):  # chowaj/pokaz top3 litery
            show_info = not show_info
        if key == ord("d"):
            print_snapshot(proba, classifier.classes_, motion)
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
