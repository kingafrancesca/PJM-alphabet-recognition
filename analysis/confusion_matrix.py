"""pelna macierz pomylek dla wszystkich 22 klas - wiersz = prawdziwa litera, kolumna = przewidziana.
na tym samym splicie co trening, dla cech xyz (obecne) i xy (bez glebi)"""
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


def load_data():
    df = pd.read_csv(CSV, sep=";")
    df = df[~df["label"].isin(DYNAMIC)]
    df["label"] = df["label"].replace(TO_BASE)
    df = df.drop_duplicates(subset=LM_COLUMNS).reset_index(drop=True)
    return df[LM_COLUMNS].values.reshape(-1, 21, 3), df["label"].values


def show(name, X, y):
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

    table = pd.DataFrame(cm, index=classes, columns=classes)
    acc = (pred == y_te).mean()
    print(f"\n=== {name}  (acc={acc:.3f}) ===")
    print("wiersz=prawdziwa, kolumna=przewidziana, kropka=0\n")
    print(table.to_string().replace(" 0", " .").replace("\t", " "))

    print("\npomylki (off-diagonal >= 1):")
    for i, true in enumerate(classes):
        errors = [(classes[j], cm[i][j]) for j in range(len(classes)) if j != i and cm[i][j] > 0]
        if errors:
            desc = ", ".join(f"{lit}x{n}" for lit, n in sorted(errors, key=lambda x: -x[1]))
            print(f"  {true} ({cm[i][i]}/{cm[i].sum()} ok)  myli z: {desc}")


def main():
    points, y = load_data()
    show("cechy xyz (obecne)", [features_from_points(p) for p in points], y)

    points_xy = points.copy()
    points_xy[:, :, 2] = 0.0
    show("cechy xy (bez glebi)", [features_from_points(p) for p in points_xy], y)


if __name__ == "__main__":
    main()
