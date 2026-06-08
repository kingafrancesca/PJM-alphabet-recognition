"""opis ruchu dloni z bufora klatek - kierunek, amplituda, zakres, liczba zawrocen.
podstawa regul typu 'A + ruch w dol = Ą' albo 'Z + dwie kropki = Ż'"""


def _hand_scale(frame):
    """dlugosc dloni: nadgarstek (lm 0) -> nasada srodkowego palca (lm 9).
    skaluje ruch - przesuniecie 0.05 znaczy wiecej dla malej dloni niz dla duzej"""
    wrist, mid = frame[0], frame[9]
    d = ((mid[0] - wrist[0]) ** 2 + (mid[1] - wrist[1]) ** 2) ** 0.5
    return d if d > 1e-6 else 1e-6


def _direction(dx, dy, threshold):
    """nazwa kierunku netto, albo 'none' jak przesuniecie ponizej progu"""
    if (dx * dx + dy * dy) ** 0.5 < threshold:
        return "none"
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _reversals(values, threshold):
    """ile razy ciag zmienia kierunek - do liczenia kropek przy Ż.
    drobne drgania ponizej progu pomijamy, zeby szum sie nie liczyl"""
    signs = []
    for a, b in zip(values, values[1:]):
        delta = b - a
        if abs(delta) < threshold:
            continue
        sign = 1 if delta > 0 else -1
        if not signs or signs[-1] != sign:
            signs.append(sign)
    return max(len(signs) - 1, 0)


def describe_motion(buffer, point=0, motion_threshold=0.15):
    """zwraca dict: direction, amplitude, range, reversals. None gdy bufor niepelny.
    wszystko skalowane dlugoscia dloni. point - ktory landmark sledzimy
    (domyslnie nadgarstek, moze byc np. czubek palca)"""
    if not buffer.is_full():
        return None

    trajectory = buffer.trajectory(point)
    scale = _hand_scale(buffer.frames[-1])
    xs = [p[0] / scale for p in trajectory]
    ys = [p[1] / scale for p in trajectory]

    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    amplitude = (dx * dx + dy * dy) ** 0.5

    range_x = max(xs) - min(xs)
    range_y = max(ys) - min(ys)
    # zawrocenia liczymy na osi, na ktorej dzialo sie najwiecej
    axis = xs if range_x > range_y else ys
    jitter_threshold = motion_threshold / 3

    return {
        "direction": _direction(dx, dy, motion_threshold),
        "amplitude": amplitude,
        "range": max(range_x, range_y),
        "reversals": _reversals(axis, jitter_threshold),
    }
