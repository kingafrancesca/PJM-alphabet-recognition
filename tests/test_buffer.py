"""Buffer: dominujaca litera z ostatnich N klatek (stabilizacja predykcji)"""

from buffer import Buffer


def test_incomplete_buffer_returns_none():
    b = Buffer(size=10, dominance_threshold=0.7)
    for _ in range(5):
        b.add("A")
    assert b.letter() is None


def test_full_unanimous_buffer_returns_letter():
    b = Buffer(size=10, dominance_threshold=0.7)
    for _ in range(10):
        b.add("A")
    assert b.letter() == "A"


def test_no_dominant_letter_returns_none():
    # 5/10 = 50% < prog 70%
    b = Buffer(size=10, dominance_threshold=0.7)
    for letter in "ABABABABAB":
        b.add(letter)
    assert b.letter() is None


def test_exactly_at_threshold_returns_letter():
    # 7/10 = 70% == prog
    b = Buffer(size=10, dominance_threshold=0.7)
    for letter in "AAAAAAABBB":
        b.add(letter)
    assert b.letter() == "A"


def test_deque_slides_with_new_letters():
    # 10x A, potem 8x B -> w buforze AABBBBBBBB (8 B) -> B
    b = Buffer(size=10, dominance_threshold=0.7)
    for _ in range(10):
        b.add("A")
    assert b.letter() == "A"
    for _ in range(8):
        b.add("B")
    assert b.letter() == "B"
