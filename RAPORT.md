# Rozpoznawanie alfabetu Polskiego Języka Migowego (PJM)

**Autor:** Kinga Kinowska
**Temat:** Rozpoznawanie liter alfabetu palcowego PJM z obrazu z kamery, w czasie rzeczywistym

---

## 1. Cel i problem

Celem projektu jest rozpoznawanie liter daktylograficznych (alfabetu palcowego) PJM
pokazywanych do kamery i odczytywanie ich na głos. Kluczowa trudność: część liter
różni się **nie samą pozą dłoni, lecz ruchem**. Litery z diakrytykami (ą, ć, ę, ł, ń,
ó, ś, ź, ż) i dwuznaki (ch, cz, rz, sz, j) to ta sama statyczna poza co litera bazowa
plus charakterystyczny gest — np. „ą” to „a” z ruchem w dół, a „ż” to „z” z dwiema
„kropkami” w powietrzu. Sam klasyfikator obrazu tego nie rozróżni.

## 2. Architektura — trzy warstwy języka

Rozwiązanie celowo odwzorowuje klasyczny podział lingwistyczny, gdzie każda warstwa
dokłada wyższy poziom interpretacji:

| Warstwa | Rola | Realizacja |
|---|---|---|
| **Syntaktyka** | wykrycie formy | MediaPipe `HandLandmarker` zwraca 21 punktów dłoni z każdej klatki |
| **Semantyka** | znaczenie pozy | Random Forest klasyfikuje punkty do jednej z 23 klas (22 litery bazowe + „sz” jako osobny, charakterystyczny kształt) |
| **Pragmatyka** | znaczenie w kontekście ruchu | heurystyka ruchu decyduje o finalnym znaku (litera bazowa + gest → ą, ż, cz…) |

Wynik trafia na ekran (z polskimi znakami) oraz do syntezatora mowy działającego
w osobnym wątku.

## 3. Dane

- Dataset bazowy: *Polish Sign Language – Google Landmarks* (Kaggle), gotowe
  współrzędne 21 punktów dłoni — bez konieczności przetwarzania surowych zdjęć.
- Po odfiltrowaniu liter czysto dynamicznych i scaleniu diakrytyków z literami
  bazowymi dataset bazowy daje **3173 próbki, 22 klasy**.
- **Personalizacja (kluczowy krok).** Model uczony wyłącznie na danych z Kaggle
  osiągał wysoką dokładność na zbiorze testowym, ale na żywo masowo mylił litery
  (o↔s, t↔f, d↔z) — klasyczny **domain shift**: dłonie ze zbioru ≠ dłoń użytkownika.
  Dlatego dodano skrypt **`collect.py`**, który nagrywa własne próbki landmarków
  użytkownika; `training.py` automatycznie dołącza je do treningu. Po dozbieraniu
  **~11 000 własnych próbek** zbiór urósł do **14 278 próbek**.
- Litery czysto dynamiczne (j, ch, cz, rz) są usuwane ze zbioru — statyczny
  klasyfikator i tak by ich nie obsłużył; powstają dopiero w warstwie pragmatyki.
- **„sz” jako osobna klasa.** W przeciwieństwie do pozostałych dwuznaków „sz” ma
  wyraźny, odrębny kształt dłoni (wyprostowana dłoń, mały palec do kamery), więc
  zamiast udawać je przez „s + ruch”, dostało własną klasę rozpoznawaną z jednej klatki.

## 4. Cechy i normalizacja

Z 21 punktów liczone jest **126 cech** na klatkę: 63 współrzędne punktów + 63 składowe
wektorów między wybranymi parami punktów (krawędzie szkieletu dłoni).

Normalizacja jest kluczowa dla działania na żywo: układ przesuwany jest do nadgarstka
i skalowany długością dłoni (nadgarstek → nasada środkowego palca). Dzięki temu cechy
**nie zależą od położenia dłoni w kadrze ani od jej odległości od kamery** — model
rozpoznaje gest niezależnie od tego, gdzie użytkownik trzyma rękę.

## 5. Model i wyniki

- **Random Forest**, 300 drzew, `class_weight="balanced"`, podział train/test 80/20
  ze stratyfikacją.
- **Dokładność: 91,7% → 97,3%** po personalizacji. Sam dataset Kaggle dawał 91,7%,
  ale na żywo zawodził; dodanie własnych próbek użytkownika podniosło dokładność na
  zbiorze testowym (2856 próbek, 23 klasy) do **97,3%** i — co ważniejsze —
  praktycznie usunęło pomyłki na żywo.

| | Litery |
|---|---|
| Przed personalizacją — najsłabsze | s (0,60), t (0,62), d (0,75), f (0,71) |
| Po personalizacji — te same litery | **t (0,96), d (0,96), s (0,92), f (0,96)** |

Personalizacja potwierdziła, że źródłem błędów był domain shift, a nie sam model:
po dołączeniu próbek użytkownika litery wcześniej mylone (t, d, g, o) trafiają niemal
bezbłędnie. Najtrudniejszą parą pozostaje **o↔s** (kształty obiektywnie podobne).

## 6. Obsługa ruchu (warstwa pragmatyki)

To autorska część projektu, wykraczająca poza zwykłą klasyfikację:

- **`motion.py`** — z bufora ostatnich klatek opisuje ruch nadgarstka czterema
  liczbami: kierunek, amplituda (przesunięcie netto), zakres (rozpiętość ruchu tam
  i z powrotem) oraz liczba zawróceń (do liczenia „kropek” przy ż/ź). Wszystko
  skalowane długością dłoni, więc progi są niezależne od wielkości dłoni w kadrze.
- **`rules.py`** — reguły „litera bazowa + opis ruchu → znak finalny”. Progi są
  **per litera**, bo zakres ruchu różni się między gestami. Pierwsza pasująca reguła
  wygrywa. Progi zostały **skalibrowane z logów `[debug]` na realnych gestach
  użytkownika**: okazało się, że poziom „bezruchu” sięga ~0,1–0,15, a właściwe gesty
  0,3–1,9, więc pierwotne progi (0,16) były zbyt niskie i powodowały „żonglowanie”
  między literą bazową a diakrytykiem. Niektóre rozróżnienia opierają się na **kierunku
  ruchu** — np. „ch” (w dół) i „cz” (w bok) dzielą ten sam kształt dłoni (bazę „h”)
  i różni je wyłącznie kierunek gestu.

## 7. Stabilizacja i wyjście

- **`buffer.py`** — predykcja z pojedynczej klatki migocze, więc finalna litera jest
  wybierana dopiero, gdy **dominuje w ≥70% z 10 ostatnich klatek**. Eliminuje to
  losowe przeskoki.
- **`speech.py`** — synteza mowy (pyttsx3, głos polski) w osobnym wątku, żeby nie
  zacinać podglądu wideo. Litera jest czytana **raz** — kolejny odczyt tej samej
  litery wymaga zabrania dłoni z kadru (reset). Kolejka mowy ma **limit**: gdy synteza
  nie nadąża za zmianami liter, zaległości są pomijane, zamiast czytać dawno
  nieaktualne znaki.

## 8. Testy

Zestaw **pytest (35 testów)** pokrywa logikę niezależną od kamery: opis ruchu, reguły
diakrytyków/dwuznaków, bufory predykcji i klatek, kolejkowanie mowy (TTS zastąpione
atrapą — testy są ciche) oraz zgodność wektorów cech i poprawność modelu na próbkach
z datasetu.

## 9. Ograniczenia

- **Z / Ź / Ż — najsłabszy punkt i granica architektury.** Te trzy znaki różni
  *liczba zawróceń* podczas rysowania (z = 0, ź = 1, ż = 2). Problem jest jednak
  fundamentalny: **samo rysowanie „Z” w powietrzu generuje zawrócenia**, a okno ruchu
  (15 klatek ≈ 0,5 s) łapie różne fragmenty gestu — w jednej chwili widzi 0 zawróceń,
  w następnej 2. Skutkuje to przeskakiwaniem z→ź→ż w trakcie jednego gestu. Podniesienie
  progu zakresu ruchu (do 0,6) ogranicza fałszywe zawrócenia, ale nie usuwa przyczyny:
  **heurystyka oparta na przesuwnym oknie nie radzi sobie z gestami wieloetapowymi**.
- **Para o↔s** — kształty statycznie podobne; mimo personalizacji to wciąż najczęstsza
  pomyłka (choć rzadka).
- **Litery dynamiczne palców** (np. wariacje f, g, k) — statyczny klasyfikator widzi
  pojedynczą klatkę, więc rozróżnia je tylko, jeśli różnią się samym kształtem dłoni.
- Synteza mowy i obsługa czcionek są zależne od Windows.

## 10. Dalszy rozwój — model sekwencyjny (LSTM)

Pełne rozwiązanie problemu liter wieloetapowych (Z/Ź/Ż) wymagałoby zastąpienia
warstwy heurystyki **modelem sekwencyjnym** uczonym na całym ruchu, a nie na
pojedynczej klatce. To jednak **przeprojektowanie**, nie poprawka — wymaga zmian na
kilku poziomach:

- **Dane (główny koszt).** Obecny zbiór to pojedyncze klatki. Model sekwencyjny uczy
  się z **serii klatek w czasie**, więc każdy gest trzeba nagrać jako sekwencję
  (np. 30 klatek × 21 punktów), w wielu powtórzeniach i najlepiej od różnych osób.
  Etykieta jest *per sekwencja*, nie per klatka — dotychczasowy CSV staje się
  bezużyteczny do trenowania ruchu.
- **Model.** Random Forest → sieć sekwencyjna (**LSTM/GRU**, ewentualnie lżejsze
  1D-CNN / TCN lub Transformer), wejście `(T_klatek, cechy)`, wyjście po pełnym
  alfabecie (diakrytyki i dwuznaki jako prawdziwe klasy). Wymaga frameworka
  PyTorch / TensorFlow zamiast scikit-learn.
- **Inferencja na żywo.** Zamiast klasyfikacji co klatkę — analiza okna klatek plus
  **segmentacja gestu** („kiedy gest się zaczyna i kończy?”), co jest osobnym, nietrywialnym
  problemem (*gesture spotting*).
- **Efekt.** Cała warstwa `motion.py` + `rules.py` (progi, kalibracja, problem Z/Ź/Ż)
  przestaje istnieć — model uczy się ruchu samodzielnie.

Wariantem pośrednim, tańszym niż pełny LSTM, jest **hybryda**: zachować obecny
klasyfikator statyczny dla póz, a mały model sekwencyjny dołożyć tylko dla garstki
gestów czysto dynamicznych. Wniosek: wąskim gardłem nie jest sam model, lecz
**dane sekwencyjne i segmentacja gestu na żywo**.

## 11. Podsumowanie

System łączy gotowy detektor punktów dłoni, własną normalizację cech, klasyfikator
uczony maszynowo i autorską warstwę interpretacji ruchu w jeden potok działający
w czasie rzeczywistym. Trójwarstwowy podział (syntaktyka → semantyka → pragmatyka)
pozwala rozpoznawać pozy bazowe (**97,3%** dokładności po personalizacji) oraz pełny
zestaw polskich diakrytyków i dwuznaków, których nie da się odróżnić z samego
nieruchomego obrazu dłoni. Kluczowym wnioskiem praktycznym okazała się
**personalizacja** — dołączenie własnych próbek użytkownika zlikwidowało domain shift,
który był główną przyczyną błędów na żywo. Granicą obecnego podejścia są gesty
wieloetapowe (Z/Ź/Ż), których pełne rozpoznanie wymagałoby modelu sekwencyjnego.
