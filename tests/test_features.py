"""sprawdza czy VECTOR_PAIRS ma poprawna kolejnosc - porownuje surowe wektory z kolumnami 0_point_vec_* w CSV"""
from pathlib import Path

import pandas as pd

from features import VECTOR_PAIRS

CSV = Path(__file__).parent / "data" / "polish sign language landmarks data.csv"


def main():
    df = pd.read_csv(CSV, sep=";")
    row = df.iloc[0]

    points = [
        [row[f"0_point_lm_{i}_x"], row[f"0_point_lm_{i}_y"], row[f"0_point_lm_{i}_z"]]
        for i in range(21)
    ]

    # surowe wektory (przed normalizacja) musza sie zgadzac z 0_point_vec_* w CSV
    vectors = []
    for a, b in VECTOR_PAIRS:
        vectors.extend([
            points[b][0] - points[a][0],
            points[b][1] - points[a][1],
            points[b][2] - points[a][2],
        ])

    vec_columns = [c for c in df.columns if c.startswith("0_point_vec_")]
    from_csv = [row[c] for c in vec_columns]

    print(f"liczba wektorow wyliczonych: {len(vectors)}")
    print(f"liczba wektorow w CSV:       {len(from_csv)}")

    max_diff = max(abs(a - b) for a, b in zip(vectors, from_csv))
    print(f"max roznica:                {max_diff:.2e}")

    if max_diff < 1e-6:
        print("OK - kolejnosc VECTOR_PAIRS zgadza sie z CSV")
    else:
        print("ROZJAZD - sprawdz kolejnosc VECTOR_PAIRS")
        for i, (a, b) in enumerate(zip(vectors, from_csv)):
            if abs(a - b) > 1e-6:
                print(f"  indeks {i} ({vec_columns[i]}): wyliczono {a:.6f}, w CSV {b:.6f}")


if __name__ == "__main__":
    main()
