from collections import deque

class FrameBuffer:
    def __init__(self, size=25):
        self.frames = deque(maxlen=size)

    def add(self, landmarks):
        if landmarks is None:
            self.frames.clear()
            return
        self.frames.append([[lm.x, lm.y, lm.z] for lm in landmarks])

    def is_full(self):
        return len(self.frames) == self.frames.maxlen

    def trajectory(self, point):
        return [(frame[point][0], frame[point][1]) for frame in self.frames]

    def displacement(self, point=0):
        if not self.is_full():
            return None
        start = self.frames[0][point]
        end = self.frames[-1][point]
        return end[0] - start[0], end[1] - start[1]
