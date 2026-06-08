"""FrameBuffer trzyma surowe klatki landmarkow i liczy przesuniecie nadgarstka"""

from types import SimpleNamespace

import pytest

from frame_buffer import FrameBuffer


def fake_hand(x=0.5, y=0.5):
    """21 landmarkow w jednym miejscu - do testu wystarczy pozycja nadgarstka"""
    return [SimpleNamespace(x=x, y=y, z=0.0) for _ in range(21)]


def test_incomplete_buffer_has_no_displacement():
    b = FrameBuffer(size=10)
    for _ in range(5):
        b.add(fake_hand())
    assert b.is_full() is False
    assert b.displacement() is None


def test_full_buffer_in_place_has_zero_displacement():
    b = FrameBuffer(size=10)
    for _ in range(10):
        b.add(fake_hand())
    assert b.is_full() is True
    dx, dy = b.displacement()
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(0.0)


def test_downward_motion_gives_positive_dy():
    b = FrameBuffer(size=10)
    for i in range(10):
        b.add(fake_hand(y=0.3 + i * 0.05))
    dx, dy = b.displacement()
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(0.45)
    assert len(b.trajectory(0)) == 10


def test_none_clears_buffer():
    b = FrameBuffer(size=10)
    for i in range(10):
        b.add(fake_hand(y=0.3 + i * 0.05))
    b.add(None)
    assert b.is_full() is False
    assert len(b.frames) == 0
