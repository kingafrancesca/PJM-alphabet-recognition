import threading
from queue import Queue, Full
import pyttsx3

try:
    import pythoncom
except ImportError:
    pythoncom = None

# syntezator mowy w osobnym watku, aby nie blokowac petli wideo
class Speech:
    def __init__(self, rate=170):
        self.queue = Queue(maxsize=3)
        self.last = None
        self.rate = rate
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def _loop(self):
        if pythoncom is not None:
            pythoncom.CoInitialize()
        while True:
            text = self.queue.get()
            print(f"[mowa] czytam: {text}")
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            for voice in engine.getProperty("voices"):
                if "polish" in voice.name.lower() or "paulina" in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    break
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
            print(f"[mowa] skonczone: {text}")

    def say(self, letter):
        if letter == self.last:
            return
        self.last = letter
        try:
            self.queue.put_nowait(letter)
        except Full:
            pass

    def reset(self):
        self.last = None
