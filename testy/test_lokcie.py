"""Łączenie łokci z dłońmi (lokcie.SledzenieLokci) na danych syntetycznych, bez kamery i bez modelu."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlonie import Dlon  # noqa: E402
from lokcie import PRZYTRZYMANIE_S, Ramie, SledzenieLokci  # noqa: E402

W, H = 1000, 1000


def dlon(id_, nadgarstek, strona="Prawa"):
    """Dłoń palcami do góry: nadgarstek w podanym miejscu, nasada środkowego palca 100 px wyżej."""
    x, y = nadgarstek
    punkty = [(x / W, y / H, 0.0)] * 21
    punkty[9] = (x / W, (y - 100) / H, 0.0)
    return Dlon(strona, 1.0, punkty, [(0.0, 0.0, 0.0)] * 21, id=id_)


def ramie(nadgarstek, lokiec, widocznosc=0.9, dz=0.05):
    return Ramie(np.array(lokiec) / [W, H], np.array(nadgarstek) / [W, H], widocznosc, dz)


s = SledzenieLokci()
lewa, prawa = dlon(1, (300, 500), "Lewa"), dlon(2, (700, 500))
ramiona = [ramie((690, 510), (720, 800)), ramie((310, 505), (250, 790))]   # celowo w odwrotnej kolejności

s.przypisz([lewa, prawa], ramiona, 0.0, 0.0, W, H)
assert lewa.lokiec is not None and prawa.lokiec is not None
assert abs(lewa.lokiec[0] * W - 250) < 1 and abs(prawa.lokiec[0] * W - 720) < 1, (lewa.lokiec, prawa.lokiec)
assert abs(prawa.lokiec[2] - 0.05) < 1e-6
print("1. każda dłoń dostaje łokieć swojej ręki:", "OK")

# Ten sam wynik modelu ciała podany kilka razy nie zmienia łokcia (filtr liczy go raz).
przed = prawa.lokiec
for t in (0.03, 0.06, 0.1):
    s.przypisz([lewa, prawa], ramiona, 0.0, t, W, H)
assert prawa.lokiec == przed
print("2. ten sam wynik użyty raz:", "OK")

# Model ciała przestaje widzieć łokcie: przez chwilę trzymamy ostatni, potem znika.
s.przypisz([lewa, prawa], [], 0.2, 0.2, W, H)
assert prawa.lokiec is not None
s.przypisz([lewa, prawa], [], 0.2, PRZYTRZYMANIE_S + 0.05, W, H)
assert prawa.lokiec is None and lewa.lokiec is None
print("3. przytrzymanie i znikanie:", "OK")

# Łokieć słabo widoczny albo nadgarstek z modelu ciała daleko od dłoni: bez łokcia.
s.przypisz([prawa], [ramie((700, 500), (720, 800), widocznosc=0.2)], 1.0, 1.0, W, H)
assert prawa.lokiec is None
s.przypisz([prawa], [ramie((950, 200), (900, 600))], 1.1, 1.1, W, H)
assert prawa.lokiec is None
print("4. odrzucanie niepewnych łokci:", "OK")

# Wygładzanie: drżenie łokcia o ±6 px jest mocno stłumione.
s = SledzenieLokci()
rng = np.random.default_rng(0)
xs = []
for i in range(60):
    t = i / 15
    s.przypisz([prawa], [ramie((700, 500), (720 + rng.normal(0, 6), 800))], t, t, W, H)
    xs.append(prawa.lokiec[0] * W)
print(f"5. drganie łokcia: surowe 6.0 px -> po wygładzeniu {np.std(np.diff(xs[10:])) / np.sqrt(2):.1f} px")
