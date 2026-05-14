"""liczy 126 cech z 21 punktow dloni - znormalizowanych wzgledem nadgarstka i skali dloni"""

# pary punktow do wektorow - kolejnosc dokladnie jak w kolumnach 0_point_vec_* w CSV
VECTOR_PAIRS = [
    (3, 4), (0, 5), (17, 18), (0, 17), (13, 14), (13, 17), (18, 19),
    (5, 6), (5, 9), (14, 15), (0, 1), (9, 10), (1, 2), (9, 13), (10, 11),
    (19, 20), (6, 7), (15, 16), (2, 3), (11, 12), (7, 8),
]


def normalize(points):
    """przesuwa uklad do nadgarstka i skaluje dlugoscia dloni (lm 0 -> lm 9),
    zeby cechy nie zalezaly od pozycji ani odleglosci dloni od kamery"""
    wrist = points[0]
    shifted = [
        [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]
        for p in points
    ]
    middle = shifted[9]
    scale = (middle[0] ** 2 + middle[1] ** 2 + middle[2] ** 2) ** 0.5
    if scale < 1e-6:
        scale = 1e-6
    return [[p[0] / scale, p[1] / scale, p[2] / scale] for p in shifted]


def features_from_points(points):
    """points: lista 21 trojek [x, y, z]. zwraca 126 cech - 63 landmarki + 63 wektory"""
    norm = normalize(points)
    features = []
    for p in norm:
        features.extend(p)
    for a, b in VECTOR_PAIRS:
        features.extend([
            norm[b][0] - norm[a][0],
            norm[b][1] - norm[a][1],
            norm[b][2] - norm[a][2],
        ])
    return features


def get_features(landmarks):
    """landmarks: 21 obiektow z polami .x, .y, .z (z MediaPipe). zwraca 126 cech"""
    points = [[lm.x, lm.y, lm.z] for lm in landmarks]
    return features_from_points(points)
