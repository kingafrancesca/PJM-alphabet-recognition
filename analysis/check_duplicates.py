"""ile unikalnych wzorcow landmarkow jest w kazdej grupie cech i per litera -
sprawdzamy, czy 0_point to faktyczne osobne 'zdjecia', czy duplikaty"""
from pathlib import Path

import pandas as pd

CSV = Path(__file__).parent.parent / "data" / "polish sign language landmarks data.csv"

df = pd.read_csv(CSV, sep=";")

features_0 = [c for c in df.columns if c.startswith("0_point_")]
features_1 = [c for c in df.columns if c.startswith("1_point_")]
features_2 = [c for c in df.columns if c.startswith("2_point_")]

print(f"unikalne wzorce w grupach cech (na {len(df)} wierszy):")
print(f"  0_point: {df[features_0].drop_duplicates().shape[0]}")
print(f"  1_point: {df[features_1].drop_duplicates().shape[0]}")
print(f"  2_point: {df[features_2].drop_duplicates().shape[0]}")

print("\nunikalne 0_point per litera:")
for label in sorted(df["label"].unique()):
    group = df[df["label"] == label]
    unique = group[features_0].drop_duplicates().shape[0]
    print(f"  {label}: {unique} unikalnych z {len(group)} wierszy")
