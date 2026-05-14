"""sanity check: czy model + get_features() poprawnie klasyfikuja probki z CSV"""
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd

from features import get_features

CSV = Path(__file__).parent / "data" / "polish sign language landmarks data.csv"
MODEL = Path(__file__).parent / "model.joblib"

DYNAMIC = {"j", "ch", "cz", "rz", "sz"}
TO_BASE = {"a+": "a", "c+": "c", "e+": "e", "l+": "l", "n+": "n",
           "o+": "o", "s+": "s", "z+": "z", "z-": "z"}


def main():
    bundle = joblib.load(MODEL)
    model = bundle["model"]

    df = pd.read_csv(CSV, sep=";")
    df = df[~df["label"].isin(DYNAMIC)].copy()
    df["true"] = df["label"].replace(TO_BASE)

    # po jednej probce na klase bazowa - zeby zobaczyc czy kazda dziala
    samples = df.groupby("true").first().reset_index()

    print(f"{'prawdziwa':>10}  {'przewidziana':>13}  {'pewnosc':>8}")
    print("-" * 40)
    ok = 0
    for _, row in samples.iterrows():
        landmarks = [SimpleNamespace(
            x=row[f"0_point_lm_{i}_x"],
            y=row[f"0_point_lm_{i}_y"],
            z=row[f"0_point_lm_{i}_z"],
        ) for i in range(21)]
        features = get_features(landmarks)
        pred = model.predict([features])[0]
        proba = model.predict_proba([features])[0].max()
        mark = "OK" if pred == row["true"] else "X"
        print(f"{row['true']:>10}  {pred:>13}  {proba:>7.2f}  {mark}")
        if pred == row["true"]:
            ok += 1

    print(f"\npoprawne: {ok}/{len(samples)}")


if __name__ == "__main__":
    main()
