from collections import deque


class FrameBuffer:
    """trzyma ostatnie surowe klatki landmarkow - do wykrywania ruchu.
    inny niz Buffer: tamten trzyma gotowe predykcje, ten cale pozycje dloni"""

    def __init__(self, size=25):
        self.frames = deque(maxlen=size)

    def add(self, landmarks):
        """landmarks: 21 obiektow z polami .x, .y, .z, albo None gdy nie ma dloni.
        brak dloni czysci bufor - przerwa rozbija ciaglosc ruchu"""
        if landmarks is None:
            self.frames.clear()
            return
        self.frames.append([[lm.x, lm.y, lm.z] for lm in landmarks])

    def is_full(self):
        return len(self.frames) == self.frames.maxlen

    def trajectory(self, point):
        """lista (x, y) danego punktu przez wszystkie klatki, od najstarszej do najnowszej"""
        return [(frame[point][0], frame[point][1]) for frame in self.frames]

    def displacement(self, point=0):
        """wektor (dx, dy) od pierwszej do ostatniej klatki. None jak bufor niepelny.
        domyslnie nadgarstek"""
        if not self.is_full():
            return None
        start = self.frames[0][point]
        end = self.frames[-1][point]
        return end[0] - start[0], end[1] - start[1]
