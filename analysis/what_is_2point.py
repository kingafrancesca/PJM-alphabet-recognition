"""co to wlasciwie jest 2_point? sprawdzamy czy to translacja, rotacja, zmiana skali
czy centrowanie 0_point - cztery testy na probie 200 wierszy"""
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).parent.parent / "data" / "polish sign language landmarks data.csv"


def landmarks(row, prefix):
    return np.array([
        [row[f"{prefix}lm_{i}_x"], row[f"{prefix}lm_{i}_y"], row[f"{prefix}lm_{i}_z"]]
        for i in range(21)
    ])


def distances(pts):
    return np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)


def main():
    df = pd.read_csv(CSV, sep=";")
    df = df.drop_duplicates(subset=[c for c in df.columns if c.startswith("0_point_")])
    df = df.sample(n=200, random_state=42).reset_index(drop=True)

    translation_std = []   # czy 2_point = 0_point + stala translacja
    max_dist_diff = []     # czy odleglosci miedzy punktami sa zachowane
    sizes_0, sizes_2 = [], []          # rozmiar bounding boxa
    centroids_0, centroids_2 = [], []

    for _, row in df.iterrows():
        p0 = landmarks(row, "0_point_")
        p2 = landmarks(row, "2_point_")

        diff = p2 - p0
        translation_std.append(diff.std(axis=0).max())

        d0 = distances(p0)
        d2 = distances(p2)
        max_dist_diff.append(np.abs(d0 - d2).max())

        sizes_0.append(np.ptp(p0[:, :2], axis=0))
        sizes_2.append(np.ptp(p2[:, :2], axis=0))

        centroids_0.append(p0.mean(axis=0))
        centroids_2.append(p2.mean(axis=0))

    print("TEST 1: czy 2_point = 0_point + stala translacja?")
    print(f"  srednia z 'max std' roznicy w wierszach: {np.mean(translation_std):.5f}")
    print("  (im wieksze, tym dalsze od czystej translacji)\n")

    print("TEST 2: czy odleglosci miedzy punktami sa zachowane?")
    print(f"  max roznica odleglosci w wierszu (srednia): {np.mean(max_dist_diff):.5f}")
    print("  (~0 to rigid transform, >0.05 to byla zmiana skali)\n")

    print("TEST 3: rozmiar bounding boxa dloni")
    print(f"  0_point sredni rozmiar (X, Y): {np.mean(sizes_0, axis=0)}")
    print(f"  2_point sredni rozmiar (X, Y): {np.mean(sizes_2, axis=0)}")
    print(f"  0_point std rozmiaru: {np.std(sizes_0, axis=0)}")
    print(f"  2_point std rozmiaru: {np.std(sizes_2, axis=0)}")
    print("  (maly std 2_point = wszystkie dlonie znormalizowane do tego samego rozmiaru)\n")

    print("TEST 4: centroidy (srednia pozycja 21 punktow)")
    print(f"  0_point sredni centroid: {np.mean(centroids_0, axis=0)}")
    print(f"  2_point sredni centroid: {np.mean(centroids_2, axis=0)}")
    print(f"  0_point std centroidu: {np.std(centroids_0, axis=0)}")
    print(f"  2_point std centroidu: {np.std(centroids_2, axis=0)}")
    print("  (maly std centroidu 2_point = wszystkie dlonie wycentrowane w to samo miejsce)")


if __name__ == "__main__":
    main()
