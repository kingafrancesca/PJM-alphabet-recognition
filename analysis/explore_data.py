"""eksploracja datasetu PJM - co tam w ogole jest, zanim zaczniemy trenowac.
sprawdza etykiety, balans klas, grupy kolumn i braki danych"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "polish sign language landmarks data.csv"


def main():
    df = pd.read_csv(DATA_PATH, sep=";")

    print(f"wierszy: {len(df)}, kolumn: {len(df.columns)}")

    print("\netykiety i liczba probek:")
    counts = df["label"].value_counts().sort_index()
    print(counts)
    print(f"klas: {df['label'].nunique()}")

    print(f"\nbalans: min {counts.min()} ({counts.idxmin()}), "
          f"max {counts.max()} ({counts.idxmax()}), srednia {counts.mean():.1f}")

    print("\ngrupy kolumn:")
    feature_cols = [c for c in df.columns if c not in ("label", "image_id")]
    for prefix in ("0_point_", "1_point_", "2_point_"):
        group = [c for c in feature_cols if c.startswith(prefix)]
        lm = [c for c in group if "_lm_" in c]
        vec = [c for c in group if "_vec_" in c]
        print(f"  {prefix}* -> {len(group)} kolumn ({len(lm)} landmarki + {len(vec)} wektory)")

    nan_total = df.isna().sum().sum()
    print(f"\nbraki danych (NaN): {nan_total}")
    if nan_total > 0:
        print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()
