"""describe_motion: kierunek, amplituda, zakres i liczba zawrocen z bufora klatek"""

from types import SimpleNamespace

import pytest

from frame_buffer import FrameBuffer
from motion import describe_motion


def fake_hand(wx, wy):
    """21 landmarkow - liczy sie nadgarstek (lm 0) i nasada srodkowego palca (lm 9).
    lm 9 dajemy 0.1 nad nadgarstkiem, zeby skala dloni byla sensowna"""
    points = [SimpleNamespace(x=wx, y=wy, z=0.0) for _ in range(21)]
    points[9] = SimpleNamespace(x=wx, y=wy - 0.1, z=0.0)
    return points


def load(buffer, positions):
    for wx, wy in positions:
        buffer.add(fake_hand(wx, wy))


def test_incomplete_buffer_returns_none():
    b = FrameBuffer(size=10)
    load(b, [(0.5, 0.5)] * 5)
    assert describe_motion(b) is None


def test_still_hand_has_no_direction():
    b = FrameBuffer(size=10)
    load(b, [(0.5, 0.5)] * 10)
    m = describe_motion(b)
    assert m["direction"] == "none"
    assert m["amplitude"] == pytest.approx(0.0)
    assert m["reversals"] == 0


def test_downward_motion():
    b = FrameBuffer(size=10)
    load(b, [(0.5, 0.3 + i * 0.03) for i in range(10)])
    m = describe_motion(b)
    assert m["direction"] == "down"
    assert m["reversals"] == 0
    assert m["amplitude"] > 0


def test_one_dot_is_single_reversal():
    # w dol i z powrotem - przesuniecie netto ~0, ale jedno zawrocenie
    b = FrameBuffer(size=10)
    load(b, [(0.5, 0.4 + d) for d in
             (0, .05, .1, .15, .2, .15, .1, .05, 0, 0)])
    m = describe_motion(b)
    assert m["direction"] == "none"
    assert m["reversals"] == 1
    assert m["range"] > 0


def test_two_dots_are_three_reversals():
    # dol-gora-dol-gora -> 3 zmiany kierunku (dwie kropki przy Z)
    b = FrameBuffer(size=10)
    load(b, [(0.5, 0.4 + d) for d in
             (0, .1, 0, .1, 0, 0, 0, 0, 0, 0)])
    m = describe_motion(b)
    assert m["reversals"] == 3
