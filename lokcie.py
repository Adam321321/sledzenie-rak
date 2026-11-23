"""
Łokcie: Google MediaPipe Pose Landmarker (model ciała, 33 punkty). Bierzemy z niego nadgarstek i łokieć
każdej ręki, żeby przedramię w widokach Anatomia/Rentgen szło od nadgarstka do łokcia, zamiast trzymać
się dłoni na sztywno (wtedy przy zgiętym nadgarstku „wyjeżdża” poza rękę).

Model ciała jest wolniejszy od modelu dłoni (na Ryzenie 7 2700 ok. 70 ms na klatkę), więc aplikacja liczy
go we własnym wątku, w swoim tempie, a łokieć jest wygładzany i przytrzymywany między wynikami.
Przedramię i tak zaczyna się w nadgarstku z modelu dłoni, aktualizowanym co klatkę.

Moduł nie zależy od okna aplikacji.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from dlonie import Dlon
from stabilizacja import FiltrOneEuro

MODEL = Path(__file__).resolve().parent / "modele" / "pose_landmarker_full.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")

RAMIONA = ((13, 15), (14, 16))   # (łokieć, nadgarstek) obu rąk w numeracji modelu ciała
MIN_WIDOCZNOSC = 0.5             # pewność modelu, że łokieć jest widoczny
MAKS_ODLEGLOSC = 1.2             # nadgarstek z modelu ciała ↔ z modelu dłoni, w rozmiarach dłoni
PRZEDRAMIE_MIN, PRZEDRAMIE_MAKS = 0.5, 6.0   # rozsądna długość przedramienia na obrazie (w rozmiarach dłoni)
PRZYTRZYMANIE_S = 0.5            # jak długo trzymać łokieć, gdy model ciała chwilowo go nie widzi
WYGLADZANIE = {"min_cutoff": 0.8, "beta": 0.02}   # One Euro (piksele, sekundy); łokieć rusza się wolniej niż palce


@dataclass
class Ramie:
    lokiec: np.ndarray      # x, y w zakresie 0..1 względem obrazu
    nadgarstek: np.ndarray  # x, y w zakresie 0..1
    widocznosc: float       # pewność, że łokieć jest widoczny
    dz: float               # głębia łokcia względem nadgarstka w metrach (+ = dalej od kamery)


def pobierz_model() -> Path:
    """Pobiera model ciała z serwera Google, jeśli go jeszcze nie ma (9,4 MB, jednorazowo)."""
    if not MODEL.exists():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        tymczasowy = MODEL.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tymczasowy)
        tymczasowy.replace(MODEL)
    return MODEL


class DetektorRamion:
    """MediaPipe Pose Landmarker w trybie wideo (jedna osoba)."""

    def __init__(self):
        opcje = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pobierz_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._model = vision.PoseLandmarker.create_from_options(opcje)
        self._ostatni_czas = -1

    def analizuj(self, rgb: np.ndarray, czas_s: float) -> list[Ramie]:
        """rgb: obraz H×W×3 (uint8, RGB), ten sam co dla modelu dłoni (także po odbiciu lustrzanym)."""
        czas = max(int(czas_s * 1000), self._ostatni_czas + 1)   # znaczniki czasu muszą rosnąć
        self._ostatni_czas = czas
        wynik = self._model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                      data=np.ascontiguousarray(rgb)), czas)
        ramiona = []
        for punkty, swiat in zip(wynik.pose_landmarks, wynik.pose_world_landmarks):
            for lo, na in RAMIONA:
                ramiona.append(Ramie(
                    lokiec=np.array([punkty[lo].x, punkty[lo].y]),
                    nadgarstek=np.array([punkty[na].x, punkty[na].y]),
                    widocznosc=float(punkty[lo].visibility),
                    dz=float(swiat[lo].z - swiat[na].z),
                ))
        return ramiona

    def zamknij(self) -> None:
        self._model.close()


@dataclass
class _Lokiec:
    filtr: FiltrOneEuro
    wartosc: np.ndarray       # x, y w pikselach, dz w mm (wygładzone)
    ostatnio: float           # czas klatki, z której był ostatni wynik modelu ciała


class SledzenieLokci:
    """Przypisuje łokcie do dłoni (po położeniu nadgarstka), wygładza je i przytrzymuje między wynikami."""

    def __init__(self):
        self._stan: dict[object, _Lokiec] = {}
        self._ostatni_wynik = -1.0
        self._surowe: dict[object, np.ndarray] = {}     # poprzedni surowy łokieć (do pomiaru skoków)
        self._zeruj_diagnostyke()

    # ---- diagnostyka (same liczby, do aplikacja.log): czemu łokieć jest albo go nie ma
    def _zeruj_diagnostyke(self) -> None:
        self.diag = {"wyniki": 0, "z_osoba": 0, "dlonie": 0, "przyjete": 0, "brak_osoby": 0,
                     "nadgarstek_daleko": 0, "lokiec_niepewny": 0, "dlugosc": 0, "klatki": 0, "klatki_z_lokciem": 0}
        self._widocznosci: list[float] = []
        self._odleglosci: list[float] = []
        self._skoki: list[float] = []

    def _diagnozuj(self, dlonie: list[Dlon], ramiona: list[Ramie], skala: np.ndarray) -> None:
        self.diag["wyniki"] += 1
        self.diag["z_osoba"] += bool(ramiona)
        for d in dlonie:
            self.diag["dlonie"] += 1
            if not ramiona:
                self.diag["brak_osoby"] += 1
                continue
            p = np.asarray(d.punkty, dtype=float)[:, :2] * skala
            rozmiar = float(np.linalg.norm(p[9] - p[0])) + 1e-3
            odl, r = min(((float(np.linalg.norm(r.nadgarstek * skala - p[0])) / rozmiar, r) for r in ramiona),
                         key=lambda x: x[0])
            dlugosc = float(np.linalg.norm(r.lokiec * skala - p[0])) / rozmiar
            self._odleglosci.append(odl)
            self._widocznosci.append(r.widocznosc)
            if odl > MAKS_ODLEGLOSC:
                self.diag["nadgarstek_daleko"] += 1
            elif r.widocznosc < MIN_WIDOCZNOSC:
                self.diag["lokiec_niepewny"] += 1
            elif not PRZEDRAMIE_MIN <= dlugosc <= PRZEDRAMIE_MAKS:
                self.diag["dlugosc"] += 1
            else:
                self.diag["przyjete"] += 1
                klucz = self._klucz(d)
                if klucz in self._surowe:
                    self._skoki.append(float(np.linalg.norm(r.lokiec * skala - self._surowe[klucz])) / rozmiar)
                self._surowe[klucz] = r.lokiec * skala

    def raport(self) -> str:
        """Podsumowanie od ostatniego raportu (i wyzerowanie liczników)."""
        g = self.diag

        def mediana(x: list[float]) -> str:
            return f"{np.median(x):.2f}" if x else "–"

        tekst = (f"wyniki modelu ciała {g['wyniki']}, z osobą {g['z_osoba']}; dłonie {g['dlonie']}: "
                 f"łokieć przyjęty {g['przyjete']}, odrzucony: brak osoby {g['brak_osoby']}, "
                 f"nadgarstek daleko {g['nadgarstek_daleko']} (odl. mediana {mediana(self._odleglosci)} rozm. dłoni), "
                 f"łokieć niepewny {g['lokiec_niepewny']} (widoczność mediana {mediana(self._widocznosci)}), "
                 f"zła długość {g['dlugosc']}; skok surowego łokcia między wynikami: mediana {mediana(self._skoki)} "
                 f"rozm. dłoni; klatki z łokciem {g['klatki_z_lokciem']}/{g['klatki']}")
        self._zeruj_diagnostyke()
        return tekst

    @staticmethod
    def _klucz(d: Dlon) -> object:
        return d.id if d.id else d.strona     # bez stabilizacji („Wył.”) dłonie nie mają numerów

    def przypisz(self, dlonie: list[Dlon], ramiona: list[Ramie] | None, t_ramion: float,
                 t: float, szer: int, wys: int) -> None:
        """
        Ustawia d.lokiec = (x, y, dz) albo None dla każdej dłoni.
        ramiona: najnowszy wynik modelu ciała (None = jeszcze żadnego), t_ramion: czas jego klatki.
        Można podawać ten sam wynik w kolejnych klatkach; filtr bierze go pod uwagę tylko raz.
        """
        skala = np.array([szer, wys], dtype=float)
        if ramiona is not None and t_ramion > self._ostatni_wynik:
            self._ostatni_wynik = t_ramion
            self._diagnozuj(dlonie, ramiona, skala)
        if ramiona is not None:
            kandydaci = [r for r in ramiona if r.widocznosc >= MIN_WIDOCZNOSC]
            pary = []
            for i, d in enumerate(dlonie):
                p = np.asarray(d.punkty, dtype=float)[:, :2] * skala
                rozmiar = float(np.linalg.norm(p[9] - p[0])) + 1e-3
                for j, r in enumerate(kandydaci):
                    odl = float(np.linalg.norm(r.nadgarstek * skala - p[0])) / rozmiar
                    dlugosc = float(np.linalg.norm(r.lokiec * skala - p[0])) / rozmiar
                    if odl <= MAKS_ODLEGLOSC and PRZEDRAMIE_MIN <= dlugosc <= PRZEDRAMIE_MAKS:
                        pary.append((odl, i, j))
            uzyte_d, uzyte_r = set(), set()
            for _odl, i, j in sorted(pary):
                if i in uzyte_d or j in uzyte_r:
                    continue
                uzyte_d.add(i)
                uzyte_r.add(j)
                d, r = dlonie[i], kandydaci[j]
                x = np.array([[*(r.lokiec * skala), r.dz * 1000.0]])
                s = self._stan.get(self._klucz(d))
                if s is None:
                    s = self._stan[self._klucz(d)] = _Lokiec(FiltrOneEuro(**WYGLADZANIE), x[0], t_ramion)
                    s.filtr(x, t_ramion)
                elif t_ramion > s.ostatnio:          # ten sam wynik modelu ciała liczymy tylko raz
                    s.wartosc = s.filtr(x, t_ramion)[0]
                    s.ostatnio = t_ramion

        obecne = set()
        for d in dlonie:
            s = self._stan.get(self._klucz(d))
            obecne.add(self._klucz(d))
            if s is not None and t - s.ostatnio <= PRZYTRZYMANIE_S:
                d.lokiec = (s.wartosc[0] / szer, s.wartosc[1] / wys, s.wartosc[2] / 1000.0)
            else:
                d.lokiec = None
            self.diag["klatki"] += 1
            self.diag["klatki_z_lokciem"] += d.lokiec is not None
        for klucz in list(self._stan):
            if klucz not in obecne or t - self._stan[klucz].ostatnio > PRZYTRZYMANIE_S:
                del self._stan[klucz]
