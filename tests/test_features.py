"""sprawdza kolejnosc VECTOR_PAIRS - surowe wektory musza sie zgadzac z kolumnami
0_point_vec_* w CSV. wymaga datasetu w data/ (pomijane, gdy go nie ma)"""

from pathlib import Path

import pytest

from features import VECTOR_PAIRS

CSV = Path(__file__).parent.parent / "data" / "polish sign language landmarks data.csv"

pytestmark = pytest.mark.skipif(not CSV.exists(), reason="brak datasetu w data/")


def test_vector_pairs_match_csv_columns():
    pd = pytest.importorskip("pandas")
    row = pd.read_csv(CSV, sep=";").iloc[0]

    points = [
        [row[f"0_point_lm_{i}_x"], row[f"0_point_lm_{i}_y"], row[f"0_point_lm_{i}_z"]]
        for i in range(21)
    ]

    # surowe wektory (przed normalizacja) liczone z VECTOR_PAIRS
    vectors = []
    for a, b in VECTOR_PAIRS:
        vectors.extend([
            points[b][0] - points[a][0],
            points[b][1] - points[a][1],
            points[b][2] - points[a][2],
        ])

    vec_columns = [c for c in row.index if c.startswith("0_point_vec_")]
    from_csv = [row[c] for c in vec_columns]

    assert len(vectors) == len(from_csv)
    max_diff = max(abs(a - b) for a, b in zip(vectors, from_csv))
    assert max_diff < 1e-6
