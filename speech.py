import threading
import time
from queue import Queue, Full

import pyttsx3

try:
    import pythoncom  # z pywin32 - SAPI5 potrzebuje COM zainicjowanego w watku
except ImportError:
    pythoncom = None


class Speech:
    """syntezator mowy w osobnym watku - nie blokuje petli wideo"""

    def __init__(self, rate=170, delay=2.0):
        # limit kolejki - gdy mowa nie nadaza, pomijamy zaleglosci zamiast je pietrzyc
        self.queue = Queue(maxsize=3)
        self.last = None
        self.last_time = 0.0
        self.delay = delay
        self.rate = rate
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def _loop(self):
        if pythoncom is not None:
            pythoncom.CoInitialize()
        while True:
            text = self.queue.get()
            print(f"[mowa] czytam: {text}")
            # init na kazda litere - inaczej runAndWait zatyka silnik po pierwszym razie
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
        # ta sama litera trzymana w kadrze - czytamy tylko raz, az sie zmieni albo reset()
        if letter == self.last:
            return
        self.last = letter
        self.last_time = time.time()
        try:
            self.queue.put_nowait(letter)
        except Full:
            pass  # mowa zostala w tyle - pomijamy, zeby nie czytac dawno nieaktualnych liter

    def reset(self):
        """dlon zeszla z kadru / predykcja niestabilna - pozwalamy przeczytac te sama litere
        jeszcze raz, gdy pokazesz ja ponownie"""
        self.last = None
