"""Speech: kolejkowanie mowy w osobnym watku + logika anty-powtorzeniowa.
silnik TTS podmieniamy na atrape - test nie odtwarza dzwieku"""

import time
import types

import pytest

import speech as speech_module


class FakeEngine:
    """udaje silnik pyttsx3 - zapisuje wypowiedziane teksty zamiast je odtwarzac"""

    def __init__(self, recorder):
        self.recorder = recorder
        self._props = {}

    def setProperty(self, key, value):
        self._props[key] = value

    def getProperty(self, key):
        if key == "voices":
            return []
        return self._props.get(key)

    def say(self, text):
        self.recorder.append(text)

    def runAndWait(self):
        pass

    def stop(self):
        pass


@pytest.fixture
def recorder(monkeypatch):
    spoken = []
    fake = types.SimpleNamespace(init=lambda: FakeEngine(spoken))
    monkeypatch.setattr(speech_module, "pyttsx3", fake)
    monkeypatch.setattr(speech_module, "pythoncom", None)
    return spoken


def wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_new_letter_is_spoken(recorder):
    s = speech_module.Speech(delay=2.0)
    s.say("A")
    assert wait_for(lambda: recorder == ["A"])


def test_two_different_letters_both_spoken(recorder):
    s = speech_module.Speech(delay=2.0)
    s.say("A")
    s.say("B")
    assert wait_for(lambda: recorder == ["A", "B"])


def test_same_letter_repeat_skipped_within_delay(recorder):
    s = speech_module.Speech(delay=2.0)
    s.say("B")
    assert wait_for(lambda: recorder == ["B"])
    s.say("B")  # ta sama litera w oknie delay -> pomijamy
    time.sleep(0.3)
    assert recorder == ["B"]


def test_say_does_not_block_main_thread(recorder):
    s = speech_module.Speech(delay=2.0)
    start = time.perf_counter()
    for _ in range(50):
        s.say("C")
    elapsed = time.perf_counter() - start
    # mowa idzie w tle - 50 wywolan musi wrocic blyskawicznie
    assert elapsed < 0.5
