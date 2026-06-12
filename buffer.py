from collections import Counter, deque

# stabilizuje predykcje z kolejnych klatek
class Buffer:
    def __init__(self, size=10, dominance_threshold=0.7):
        self.buffer = deque(maxlen=size)
        self.threshold = dominance_threshold

    def add(self, letter):
        self.buffer.append(letter)

    def clear(self):
        # czyszczenie buffera, gdy dloni nie ma w kadrze
        self.buffer.clear()

    def letter(self):
        # litera dominujaca w buforze
        if len(self.buffer) < self.buffer.maxlen:
            return None
        counts = Counter(self.buffer)
        letter, count = counts.most_common(1)[0]
        if count / len(self.buffer) >= self.threshold:
            return letter
        return None
