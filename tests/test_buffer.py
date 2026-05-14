from buffer import Buffer


def main():
    b = Buffer(size=10, dominance_threshold=0.7)

    # mniej niz pelny bufor -> None
    for _ in range(5):
        b.add("A")
    print(f"po 5x A (bufor niepelny): {b.letter()}")

    # pelny bufor, 100% A -> "A"
    for _ in range(5):
        b.add("A")
    print(f"po 10x A: {b.letter()}")

    # mieszanka 50/50 -> None, brak dominacji 70%
    b2 = Buffer(size=10, dominance_threshold=0.7)
    for letter in "ABABABABAB":
        b2.add(letter)
    print(f"po ABABABABAB: {b2.letter()}")

    # 7 z 10 -> dominacja
    b3 = Buffer(size=10, dominance_threshold=0.7)
    for letter in "AAAAAAABBB":
        b3.add(letter)
    print(f"po AAAAAAABBB: {b3.letter()}")

    # przejscie z A na B - bufor sie przesuwa (deque)
    b4 = Buffer(size=10, dominance_threshold=0.7)
    for _ in range(10):
        b4.add("A")
    print(f"po 10x A: {b4.letter()}")
    for _ in range(8):
        b4.add("B")
    print(f"+ 8x B (w buforze AABBBBBBBB): {b4.letter()}")


if __name__ == "__main__":
    main()
