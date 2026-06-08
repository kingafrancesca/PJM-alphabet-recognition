"""porownanie modeli i zestawow cech - SVM vs Random Forest, rozne grupy 0/1/2_point"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

CSV = Path(__file__).parent.parent / "data" / "polish sign language landmarks data.csv"

DYNAMIC = {"j", "ch", "cz", "rz", "sz"}
TO_BASE = {"a+": "a", "c+": "c", "e+": "e", "l+": "l", "n+": "n",
           "o+": "o", "s+": "s", "z+": "z", "z-": "z"}


def prepare(df_raw, prefixes):
    df = df_raw[~df_raw["label"].isin(DYNAMIC)].copy()
    df["label"] = df["label"].replace(TO_BASE)
    features = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    # dedup po 0_point - to faktyczne 'zdjecia'
    features_0 = [c for c in df.columns if c.startswith("0_point_")]
    df = df.drop_duplicates(subset=features_0).reset_index(drop=True)
    return df[features].values, df["label"].values


def run(name, X, y, model):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    print(f"{name:40s} acc={acc:.3f}  cech={X.shape[1]}")
    return acc


def main():
    df_raw = pd.read_csv(CSV, sep=";")

    configs = [
        ("0_point",         ["0_point_"]),
        ("0_point+2_point", ["0_point_", "2_point_"]),
        ("wszystkie",       ["0_point_", "1_point_", "2_point_"]),
    ]

    print("SVM (RBF, C=10, balanced):")
    for name, prefixes in configs:
        X, y = prepare(df_raw, prefixes)
        run(name, X, y, SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))

    print("\nRandom Forest (300 drzew):")
    for name, prefixes in configs:
        X, y = prepare(df_raw, prefixes)
        run(name, X, y, RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1))


if __name__ == "__main__":
    main()
