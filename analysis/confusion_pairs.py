"""ktore litery model myli ze soba - i czy winna jest wspolrzedna z (glebia z MediaPipe).
macierz pomylek na tym samym splicie co trening, dla cech xyz i samych xy"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from features import features_from_points

CSV = ROOT / "data" / "polish sign language landmarks data.csv"
LM_COLUMNS = [f"0_point_lm_{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]

DYNAMIC = {"j", "ch", "cz", "rz", "sz"}
TO_BASE = {"a+": "a", "c+": "c", "e+": "e", "l+": "l", "n+": "n",
           "o+": "o", "s+": "s", "z+": "z", "z-": "z"}

# pary mylone na zywo - chcemy zobaczyc twarde liczby
SUSPECT_PAIRS = [("o", "s"), ("d", "z"), ("t", "f"), ("g", "k"), ("g", "h")]


def load_data():
    df = pd.read_csv(CSV, sep=";")
    df = df[~df["label"].isin(DYNAMIC)]
    df["label"] = df["label"].replace(TO_BASE)
    df = df.drop_duplicates(subset=LM_COLUMNS).reset_index(drop=True)
    points = df[LM_COLUMNS].values.reshape(-1, 21, 3)
    return points, df["label"].values


def train_and_check(name, X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    classes = list(model.classes_)
    cm = confusion_matrix(y_te, pred, labels=classes)
    acc = (pred == y_te).mean()
    print(f"\n--- {name}  (acc={acc:.3f}) ---")
    for a, b in SUSPECT_PAIRS:
        ia, ib = classes.index(a), classes.index(b)
        # ile razy 'a' poszlo jako 'b' i odwrotnie, na tle ilu probek
        sa, sb = cm[ia].sum(), cm[ib].sum()
        print(f"  {a}->{b}: {cm[ia][ib]}/{sa}    {b}->{a}: {cm[ib][ia]}/{sb}"
              f"    (poprawne {a}={cm[ia][ia]}/{sa}, {b}={cm[ib][ib]}/{sb})")


def main():
    points, y = load_data()

    X_xyz = [features_from_points(p) for p in points]
    train_and_check("cechy xyz (obecne)", X_xyz, y)

    # te same cechy z wyzerowanym z - czy glebia pomaga czy szkodzi
    points_xy = points.copy()
    points_xy[:, :, 2] = 0.0
    X_xy = [features_from_points(p) for p in points_xy]
    train_and_check("cechy xy (z wyzerowane)", X_xy, y)


if __name__ == "__main__":
    main()
