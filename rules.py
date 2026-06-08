"""litera bazowa z klasyfikatora + opis ruchu -> finalny znak.
np. 'a' + ruch w dol = 'ą', 'z' + dwie kropki = 'ż'.
progi sa per litera, bo zakres ruchu rozni sie miedzy gestami - kalibrowane na zywo z logow.

dla kazdej litery: lista (warunek, wynik), pierwszy pasujacy warunek wygrywa.
warunek to dict, wszystkie klucze musza pasowac naraz:
  direction      - dokladny kierunek ruchu
  min_amplitude  - min amplituda przesuniecia netto
  min_range      - min rozpietosc ruchu (lapie ruch tam i z powrotem)
  min_reversals  - min liczba zmian kierunku (kropki przy Z)"""

# progi skalibrowane 2026-06-08 z logow [debug]: poziom szumu ("bezruch") siegal ~0.1-0.15,
# realne ruchy 0.3-1.9, wiec progi ~1.5x szumu danej litery - blokuje zonglowanie baza<->diakrytyk
RULES = {
    # s nieruchome ~0.09 -> prog 0.15. sz ma wlasna klase (nie powstaje z s+ruch)
    "s": [({"min_amplitude": 0.15}, "ś")],
    # h, ch, cz dziela ten sam ksztalt dloni (baza h) - rozni je KIERUNEK ruchu.
    # ch w dol ~0.52 (szum-dol 0.21 -> prog 0.30); cz w bok ~0.27-0.63 (szum-bok 0.18 -> prog 0.25)
    "h": [({"direction": "down", "min_amplitude": 0.30}, "ch"),
          ({"direction": "left", "min_amplitude": 0.25}, "cz"),
          ({"direction": "right", "min_amplitude": 0.25}, "cz")],
    # ą ruch w dol ~1.16 -> prog 0.25 z kierunkiem (chroni przed bocznym dryfem)
    "a": [({"direction": "down", "min_amplitude": 0.25}, "ą")],
    # cz przeniesione pod baze h - tu zostaje tylko ć (szum ~0.08 -> prog 0.15)
    "c": [({"min_amplitude": 0.15}, "ć")],
    "e": [({"min_amplitude": 0.18}, "ę")],   # szum e wysoki ~0.145 -> prog 0.18
    "i": [({"direction": "down", "min_amplitude": 0.25}, "j")],
    "l": [({"min_amplitude": 0.16}, "ł")],   # szum ~0.12
    "n": [({"min_amplitude": 0.18}, "ń")],   # szum n wysoki ~0.136 -> prog 0.18
    "o": [({"min_amplitude": 0.15}, "ó")],   # szum ~0.09
    "r": [({"min_amplitude": 0.34}, "rz")],  # bez snapshotow - prog wstepny
    # z/ź/ż rozni LICZBA zawrocen (z=0, ź=1, ż=2); zakres ruchu zawsze duzy >0.5 -> prog 0.6.
    # UWAGA: to najslabsze rozroznienie - rysowanie Z samo generuje zawrocenia (patrz RAPORT)
    "z": [({"min_reversals": 2, "min_range": 0.6}, "ż"),
          ({"min_reversals": 1, "min_range": 0.6}, "ź")],
}


def _matches(condition, motion):
    if "direction" in condition and motion["direction"] != condition["direction"]:
        return False
    if "min_amplitude" in condition and motion["amplitude"] < condition["min_amplitude"]:
        return False
    if "min_range" in condition and motion["range"] < condition["min_range"]:
        return False
    if "min_reversals" in condition and motion["reversals"] < condition["min_reversals"]:
        return False
    return True


def apply_rules(base_letter, motion):
    """base_letter: znak z klasyfikatora. motion: wynik describe_motion() albo None.
    zwraca finalny znak - z diakrytykiem/dwuznakiem jak ruch pasuje, inaczej litere bazowa"""
    if motion is None:
        return base_letter
    for condition, result in RULES.get(base_letter, []):
        if _matches(condition, motion):
            return result
    return base_letter
