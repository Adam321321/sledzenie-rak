"""
Stabilizacja śledzenia dłoni (nakładka na surowe wyniki MediaPipe).

Trzy rzeczy:
1. Wygładzanie filtrem One Euro (Casiez, Roussel, Vogel, CHI 2012): gdy dłoń stoi, mocno tłumi
   drżenie punktów; gdy szybko się rusza, filtr sam się „rozluźnia”, więc prawie nie ma opóźnienia.
2. Tożsamość dłoni: dopasowanie dłoni między klatkami po położeniu. MediaPipe nie gwarantuje
   kolejności dłoni, a w aplikacji dwa modele liczą na zmianę klatki. Podpis Lewa/Prawa
   ustalany jest z ostatnich ~0,4 s, więc nie przeskakuje.
3. Ochrona przed miganiem: nowa dłoń pojawia się po 2 klatkach z rzędu, a zgubiona na moment
   nie znika od razu, tylko jest przytrzymywana (z wygaszaniem) przez ~0,15 s.

Moduł nie zależy od okna aplikacji.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from dlonie import Dlon

# Ustawienia filtru dla każdego poziomu wygładzania (jednostki: piksele i sekundy).
# min_cutoff: niżej = spokojniej w bezruchu; beta: wyżej = mniejsze opóźnienie w ruchu.
# Dobrane pomiarami (zdjęcie testowe + szum kamery, 49 kombinacji): dla każdego poziomu spokoju
# wybrane ustawienie z najmniejszym odstawaniem w szybkim ruchu. Wyniki w README („Stabilność”).
POZIOMY = {
    "slabe": {"min_cutoff": 1.0, "beta": 0.3},
    "srednie": {"min_cutoff": 0.5, "beta": 0.08},
    "mocne": {"min_cutoff": 0.3, "beta": 0.04},
}
NAZWY_POZIOMOW = {"wyl": "Wył.", "slabe": "Słabe", "srednie": "Średnie", "mocne": "Mocne"}

PRZYTRZYMANIE_S = 0.15     # jak długo pokazywać zgubioną dłoń
POTWIERDZENIE = 2          # ile klatek z rzędu, zanim pokażemy nową dłoń
MAKS_ODLEGLOSC = 2.0       # dopasowanie: maks. przesunięcie między klatkami (w rozmiarach dłoni)
GLOSY = 12                 # z ilu ostatnich klatek ustalamy stronę dłoni


class FiltrOneEuro:
    """Filtr One Euro dla tablicy punktów (N×3). Prędkość liczona wspólnie dla x i y każdego punktu."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t = 0.0

    @staticmethod
    def _alfa(cutoff, dt: float):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(1e-3, t - self._t)
        self._t = t
        a_d = self._alfa(self.d_cutoff, dt)
        self._dx = a_d * (x - self._x) / dt + (1.0 - a_d) * self._dx
        predkosc = np.linalg.norm(self._dx[:, :2], axis=1, keepdims=True)
        a = self._alfa(self.min_cutoff + self.beta * predkosc, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()


@dataclass
class _Sledzona:
    id: int
    filtr: FiltrOneEuro
    filtr_3d: FiltrOneEuro
    punkty: np.ndarray            # wygładzone, w pikselach (21×3)
    punkty_3d: np.ndarray         # wygładzone, w milimetrach (21×3)
    srodek: np.ndarray            # środek dłoni w pikselach (do dopasowania)
    rozmiar: float                # nadgarstek–nasada środkowego palca, w pikselach
    ostatnio: float               # kiedy ostatnio wykryta
    trafienia: int = 1            # ile klatek z rzędu wykryta
    glosy: deque = field(default_factory=lambda: deque(maxlen=GLOSY))


def _geometria(p: np.ndarray) -> tuple[np.ndarray, float]:
    srodek = p[[0, 5, 9, 13, 17], :2].mean(axis=0)
    rozmiar = float(np.linalg.norm(p[0, :2] - p[9, :2])) + 1e-3
    return srodek, rozmiar


class StabilizatorDloni:
    def __init__(self, poziom: str = "srednie"):
        self.poziom = poziom
        self._dlonie: list[_Sledzona] = []
        self._nastepny_id = 1

    def aktualizuj(self, dlonie: list[Dlon], t: float, szer: int, wys: int) -> list[Dlon]:
        """dlonie: surowy wynik z modelu dla jednej klatki; t: czas klatki w sekundach."""
        if self.poziom == "wyl":      # porównanie „przed”: surowe wyniki bez żadnej obróbki
            self._dlonie.clear()
            return dlonie
        ustawienia = POZIOMY[self.poziom]
        skala = np.array([szer, wys, szer], dtype=float)  # z w MediaPipe ma skalę zbliżoną do x

        wykryte = []
        for d in dlonie:
            p = np.asarray(d.punkty, dtype=float) * skala
            srodek, rozmiar = _geometria(p)
            wykryte.append((d, p, np.asarray(d.punkty_3d, dtype=float) * 1000.0, srodek, rozmiar))

        # Dopasowanie: najbliższe pary (odległość mierzona w rozmiarach dłoni).
        pary = sorted(
            (float(np.linalg.norm(s.srodek - w[3])) / max(s.rozmiar, w[4]), i, j)
            for i, s in enumerate(self._dlonie) for j, w in enumerate(wykryte)
        )
        uzyte_s, uzyte_w = set(), set()
        for koszt, i, j in pary:
            if koszt > MAKS_ODLEGLOSC or i in uzyte_s or j in uzyte_w:
                continue
            uzyte_s.add(i)
            uzyte_w.add(j)
            s = self._dlonie[i]
            d, p, p3, srodek, rozmiar = wykryte[j]
            for f in (s.filtr, s.filtr_3d):
                f.min_cutoff, f.beta = ustawienia["min_cutoff"], ustawienia["beta"]
            s.punkty = s.filtr(p, t)
            s.punkty_3d = s.filtr_3d(p3, t)
            s.srodek, s.rozmiar = _geometria(s.punkty)
            s.trafienia = s.trafienia + 1 if t - s.ostatnio <= PRZYTRZYMANIE_S else 1
            s.ostatnio = t
            s.glosy.append((d.strona, d.pewnosc))

        for j, (d, p, p3, srodek, rozmiar) in enumerate(wykryte):
            if j in uzyte_w:
                continue
            s = _Sledzona(
                id=self._nastepny_id,
                filtr=FiltrOneEuro(**ustawienia), filtr_3d=FiltrOneEuro(**ustawienia),
                punkty=p, punkty_3d=p3, srodek=srodek, rozmiar=rozmiar, ostatnio=t,
            )
            s.filtr(p, t)
            s.filtr_3d(p3, t)
            s.glosy.append((d.strona, d.pewnosc))
            self._nastepny_id += 1
            self._dlonie.append(s)

        # Zgubione dłonie: przytrzymaj chwilę, potem usuń.
        self._dlonie = [s for s in self._dlonie if t - s.ostatnio <= PRZYTRZYMANIE_S]

        wynik = []
        for s in self._dlonie:
            if s.trafienia < POTWIERDZENIE:
                continue
            suma: dict[str, float] = {}
            for strona, pewnosc in s.glosy:
                suma[strona] = suma.get(strona, 0.0) + pewnosc
            strona = max(suma, key=suma.get)
            pewnosci = [p for st, p in s.glosy if st == strona]
            wynik.append(Dlon(
                strona=strona,
                pewnosc=sum(pewnosci) / len(pewnosci),
                punkty=[tuple(v) for v in (s.punkty / skala).tolist()],
                punkty_3d=[tuple(v) for v in (s.punkty_3d / 1000.0).tolist()],
                id=s.id,
                widocznosc=1.0 if s.ostatnio == t else max(0.0, 1.0 - (t - s.ostatnio) / PRZYTRZYMANIE_S),
            ))
        return wynik
