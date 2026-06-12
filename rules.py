# dla kazdej litery: lista (warunek, wynik), pierwszy pasujacy warunek wygrywa.
# warunek to opis ruchu; wszystkie klucze musza pasowac naraz:
# direction - dokladny kierunek ruchu
# min_amplitude - min amplituda przesuniecia netto
# min_range - min rozpietosc ruchu (lapie ruch tam i z powrotem)
# min_reversals - min liczba zmian kierunku

RULES = {
    # ś to s + ruch w dol
    "s": [({"direction": "down", "min_amplitude": 0.20}, "ś")],
    # h, ch, cz dziela ten sam ksztalt dloni (baza h) - rozni je kierunek ruchu.
    "h": [({"direction": "down", "min_amplitude": 0.30}, "ch"),
          ({"direction": "left", "min_amplitude": 0.25}, "cz"),
          ({"direction": "right", "min_amplitude": 0.25}, "cz")],
    # ą = a + okrezny ruch w dol
    "a": [({"direction": "down", "min_amplitude": 0.25}, "ą")],
    # ć = c + ruch w dol
    "c": [({"direction": "down", "min_amplitude": 0.25}, "ć")],
    # ę = e + okrezny ruch w dol
    "e": [({"direction": "down", "min_amplitude": 0.25}, "ę")],
    # j = i + ruch rysujacy ogonek w dol
    "i": [({"direction": "down", "min_amplitude": 0.20}, "j")],
    # ł = l + ruch w dol
    "l": [({"direction": "down", "min_amplitude": 0.20}, "ł")],
    # ń = n + ruch w dol
    "n": [({"direction": "down", "min_amplitude": 0.20}, "ń")],
    # ó = o + ruch w dol
    "o": [({"direction": "down", "min_amplitude": 0.20}, "ó")],
    # rz = r + ruch w dol
    "r": [({"direction": "down", "min_amplitude": 0.40}, "rz")],
    # ź = z + ruch w dol, ż = z + ruch okrezny wracajacy do startu
    "z": [({"min_range": 0.55, "max_amplitude_ratio": 0.6}, "ż"),
          ({"direction": "down", "min_amplitude": 0.75, "_buffer": "short"}, "ź")],
}


def _matches(condition, motion):
    if "direction" in condition and motion["direction"] != condition["direction"]:
        return False
    if "min_amplitude" in condition and motion["amplitude"] < condition["min_amplitude"]:
        return False
    if "max_amplitude" in condition and motion["amplitude"] > condition["max_amplitude"]:
        return False
    if "max_amplitude_ratio" in condition:
        rng = motion["range"]
        if rng <= 1e-6 or motion["amplitude"] > condition["max_amplitude_ratio"] * rng:
            return False
    if "min_range" in condition and motion["range"] < condition["min_range"]:
        return False
    if "min_reversals" in condition and motion["reversals"] < condition["min_reversals"]:
        return False
    return True


def apply_rules(base_letter, motion, z_motion=None):
    long_motion = z_motion if (base_letter == "z" and z_motion is not None) else motion
    for condition, result in RULES.get(base_letter, []):
        used = motion if condition.get("_buffer") == "short" else long_motion
        if used is not None and _matches(condition, used):
            return result
    return base_letter
