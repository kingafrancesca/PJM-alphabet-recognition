from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from features import features_from_points

CSV = Path(__file__).parent / "data" / "polish sign language landmarks data.csv"
MODEL_OUT = Path(__file__).parent / "model.joblib"

LM_COLUMNS = [f"0_point_lm_{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]

# litery dynamiczne - wymagaja ruchu, statyczny klasyfikator ich nie obsluzy
DYNAMIC = {"j", "ch", "cz", "rz", "sz"}
# diakrytyki zlaczamy z litera bazowa - roznica jest w ruchu, nie w samej pozie
TO_BASE = {"a+": "a", "c+": "c", "e+": "e", "l+": "l", "n+": "n",
           "o+": "o", "s+": "s", "z+": "z", "z-": "z"}


def main():
    df = pd.read_csv(CSV, sep=";")
    df = df[~df["label"].isin(DYNAMIC)]
    df["label"] = df["label"].replace(TO_BASE)

    df = df.drop_duplicates(subset=LM_COLUMNS).reset_index(drop=True)

    # surowe landmarki -> 21 punktow na wiersz -> znormalizowane cechy
    points = df[LM_COLUMNS].values.reshape(-1, 21, 3)
    X = [features_from_points(p) for p in points]
    y = df["label"].values
    print(f"probki: {len(X)}, klasy: {len(set(y))}, cechy: {len(X[0])}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    print(f"\naccuracy: {model.score(X_te, y_te):.3f}\n")
    print(classification_report(y_te, pred, zero_division=0))

    joblib.dump({"model": model, "columns": LM_COLUMNS}, MODEL_OUT)
    print(f"zapisano model: {MODEL_OUT.name}")


if __name__ == "__main__":
    main()
