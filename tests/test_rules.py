"""regula: litera bazowa + opis ruchu -> finalny znak (diakrytyk/dwuznak)"""

from rules import apply_rules


def motion(direction="none", amplitude=0.0, range_=0.0, reversals=0):
    return {"direction": direction, "amplitude": amplitude,
            "range": range_, "reversals": reversals}


def test_a_down_becomes_a_ogonek():
    assert apply_rules("a", motion("down", amplitude=0.3)) == "ą"


def test_a_no_motion_stays_a():
    assert apply_rules("a", motion()) == "a"


def test_a_wrong_direction_stays_a():
    # ogonek wymaga ruchu w dol - ruch w bok nie lapie reguly
    assert apply_rules("a", motion("right", amplitude=0.3)) == "a"


def test_c_small_motion_is_cacute():
    # ruch od 0.16 -> ć
    assert apply_rules("c", motion("down", amplitude=0.2)) == "ć"


def test_c_large_motion_still_cacute():
    # cz przeniesione pod baze h (dzieli z nia ksztalt) - c przy duzym ruchu zostaje ć
    assert apply_rules("c", motion("down", amplitude=0.5)) == "ć"


def test_z_one_reversal_is_zacute():
    # zakres ruchu z/ź/ż zawsze duzy (>0.6) - rozroznia liczba zawrocen
    assert apply_rules("z", motion(range_=0.7, reversals=1)) == "ź"


def test_z_three_reversals_is_zdot():
    assert apply_rules("z", motion(range_=0.7, reversals=3)) == "ż"


def test_z_no_motion_stays_z():
    assert apply_rules("z", motion()) == "z"


def test_i_down_becomes_j():
    assert apply_rules("i", motion("down", amplitude=0.3)) == "j"


def test_s_thresholds():
    # s: nieruchome -> s, ruch -> ś. sz ma wlasna klase (nie powstaje z s+ruch)
    assert apply_rules("s", motion(amplitude=0.05)) == "s"
    assert apply_rules("s", motion(amplitude=0.2)) == "ś"
    assert apply_rules("s", motion(amplitude=0.4)) == "ś"


def test_sz_is_own_class_passes_through():
    # sz rozpoznawane bezposrednio przez klasyfikator - regula go nie zmienia
    assert apply_rules("sz", motion("left", amplitude=0.5)) == "sz"


def test_h_down_is_ch():
    # baza h: brak ruchu -> h, ruch w dol -> ch
    assert apply_rules("h", motion(amplitude=0.05)) == "h"
    assert apply_rules("h", motion("down", amplitude=0.4)) == "ch"


def test_h_sideways_is_cz():
    # cz dzieli ksztalt z ch (baza h), rozni je kierunek - ruch w bok -> cz
    assert apply_rules("h", motion("left", amplitude=0.4)) == "cz"
    assert apply_rules("h", motion("right", amplitude=0.4)) == "cz"


def test_letter_without_rule_passes_through():
    assert apply_rules("b", motion("down", amplitude=0.5)) == "b"


def test_motion_none_returns_base_letter():
    # bufor niepelny (motion None) -> zawsze litera bazowa
    assert apply_rules("a", None) == "a"
