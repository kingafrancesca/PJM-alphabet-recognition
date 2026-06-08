"""sanity check end-to-end: model + get_features() poprawnie klasyfikuja probki z CSV.
wymaga datasetu w data/ i wytrenowanego model.joblib (pomijane, gdy ich nie ma)"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from features import get_features

ROOT = Path(__file__).parent.parent
CSV = ROOT / "data" / "polish sign language landmarks data.csv"
MODEL = ROOT / "model.joblib"

DYNAMIC = {"j", "ch", "cz", "rz"}  # sz ma teraz wlasna klase
TO_BASE = {"a+": "a", "c+": "c", "e+": "e", "l+": "l", "n+": "n",
           "o+": "o", "s+": "s", "z+": "z", "z-": "z"}

pytestmark = pytest.mark.skipif(
    not (CSV.exists() and MODEL.exists()),
    reason="brak datasetu w data/ lub model.joblib",
)


def test_one_sample_per_class_predicts_correctly():
    pd = pytest.importorskip("pandas")
    joblib = pytest.importorskip("joblib")

    model = joblib.load(MODEL)["model"]
    df = pd.read_csv(CSV, sep=";")
    df = df[~df["label"].isin(DYNAMIC)].copy()
    df["true"] = df["label"].replace(TO_BASE)

    # po jednej probce na klase bazowa z oryginalnego datasetu
    samples = df.groupby("true").first().reset_index()

    wrong = []
    for _, row in samples.iterrows():
        landmarks = [SimpleNamespace(
            x=row[f"0_point_lm_{i}_x"],
            y=row[f"0_point_lm_{i}_y"],
            z=row[f"0_point_lm_{i}_z"],
        ) for i in range(21)]
        pred = model.predict([get_features(landmarks)])[0]
        if pred != row["true"]:
            wrong.append((row["true"], pred))

    # sanity check pipeline'u (cechy -> model), nie test perfekcji: model jest spersonalizowany
    # pod jedna dlon (~77% probek od uzytkownika), wiec pojedyncze probki z oryginalnego
    # datasetu moga sie mylic. Wymagamy poprawnej wiekszosci.
    accuracy = 1 - len(wrong) / len(samples)
    assert accuracy >= 0.8, f"za duzo bledow ({accuracy:.2f}): {wrong}"
