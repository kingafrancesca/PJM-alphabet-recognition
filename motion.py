# opis ruchu dloni z bufora klatek - kierunek, amplituda, zakres, liczba zawrocen

def _hand_scale(frame):
    # obliczanie dlugosci dloni
    wrist, mid = frame[0], frame[9]
    d = ((mid[0] - wrist[0]) ** 2 + (mid[1] - wrist[1]) ** 2) ** 0.5
    return d if d > 1e-6 else 1e-6

def _direction(dx, dy, threshold):
    if (dx * dx + dy * dy) ** 0.5 < threshold:
        return "none"
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"

def _reversals(values, threshold):
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
    # zwraca opis ruchu dloni: direction, amplitude, range, reversals. None gdy bufor niepelny.
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
    axis = xs if range_x > range_y else ys
    jitter_threshold = motion_threshold / 3

    return {
        "direction": _direction(dx, dy, motion_threshold),
        "amplitude": amplitude,
        "range": max(range_x, range_y),
        "reversals": _reversals(axis, jitter_threshold),
    }
