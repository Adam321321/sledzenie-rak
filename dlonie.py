"""
Śledzenie dłoni: Google MediaPipe Hand Landmarker (21 punktów na dłoń).

Ten moduł nie zależy od okna aplikacji, więc da się go użyć gdzie indziej (np. do wysyłania
pozycji palców do Unity).

Numery punktów (takie same jak w dokumentacji MediaPipe):
    0 nadgarstek
    1–4  kciuk          (1 = u nasady … 4 = czubek)
    5–8  wskazujący
    9–12 środkowy
    13–16 serdeczny
    17–20 mały
"""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).resolve().parent / "modele" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

NAZWY_PUNKTOW = [
    "nadgarstek",
    "kciuk 1", "kciuk 2", "kciuk 3", "kciuk – czubek",
    "wskazujący 1", "wskazujący 2", "wskazujący 3", "wskazujący – czubek",
    "środkowy 1", "środkowy 2", "środkowy 3", "środkowy – czubek",
    "serdeczny 1", "serdeczny 2", "serdeczny 3", "serdeczny – czubek",
    "mały 1", "mały 2", "mały 3", "mały – czubek",
]

# Połączenia (kości) pogrupowane tak, żeby każdy palec dało się narysować swoim kolorem.
KOSCI = {
    "dlon": [(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)],
    "kciuk": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "wskazujacy": [(5, 6), (6, 7), (7, 8)],
    "srodkowy": [(9, 10), (10, 11), (11, 12)],
    "serdeczny": [(13, 14), (14, 15), (15, 16)],
    "maly": [(17, 18), (18, 19), (19, 20)],
}


def palec_punktu(i: int) -> str:
    """Do którego palca należy punkt o numerze i."""
    if i == 0:
        return "dlon"
    return ("kciuk", "wskazujacy", "srodkowy", "serdeczny", "maly")[(i - 1) // 4]


@dataclass
class Dlon:
    strona: str                                  # "Lewa" albo "Prawa" (z punktu widzenia osoby w kadrze)
    pewnosc: float                               # 0..1, jak bardzo model jest pewny strony
    punkty: list[tuple[float, float, float]]     # 21 punktów: x, y w zakresie 0..1 względem obrazu; z = względna głębia
    punkty_3d: list[tuple[float, float, float]]  # te same punkty w metrach, względem środka dłoni
    id: int = 0                                  # numer śledzonej dłoni (nadaje stabilizacja.py)
    widocznosc: float = 1.0                      # 1 = wykryta teraz; mniej = chwilowo zgubiona (wygaszanie)
    lokiec: tuple[float, float, float] | None = None  # łokieć tej ręki (x, y jak punkty; głębia względem
                                                      # nadgarstka w metrach); ustawia lokcie.py, gdy widać


def pobierz_model(postep=None) -> Path:
    """Pobiera model z serwera Google, jeśli go jeszcze nie ma (7,8 MB, jednorazowo)."""
    if MODEL.exists():
        return MODEL
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    tymczasowy = MODEL.with_suffix(".part")

    def raport(bloki, rozmiar_bloku, calosc):
        if postep and calosc > 0:
            postep(min(1.0, bloki * rozmiar_bloku / calosc))

    urllib.request.urlretrieve(MODEL_URL, tymczasowy, raport)
    tymczasowy.replace(MODEL)
    return MODEL


class DetektorDloni:
    """
    Opakowanie na MediaPipe Hand Landmarker w trybie wideo: model śledzi dłoń z klatki na klatkę,
    zamiast szukać jej od zera, więc działa szybciej i stabilniej.
    """

    def __init__(self, liczba_dloni: int = 2, pewnosc_wykrycia: float = 0.5, pewnosc_sledzenia: float = 0.5):
        opcje = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pobierz_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=liczba_dloni,
            min_hand_detection_confidence=pewnosc_wykrycia,
            min_hand_presence_confidence=pewnosc_wykrycia,
            min_tracking_confidence=pewnosc_sledzenia,
        )
        self._model = vision.HandLandmarker.create_from_options(opcje)
        self._start = time.perf_counter()
        self._ostatni_czas = -1

    def analizuj(self, rgb: np.ndarray, lustro: bool) -> list[Dlon]:
        """
        rgb: obraz H×W×3 (uint8, RGB).
        lustro: czy obraz jest lustrzanym odbiciem (jak selfie). MediaPipe 1.x podaje stronę dłoni
        dla zwykłego (nieodbitego) obrazu z kamery, więc przy odbiciu zamieniamy strony.
        (Uwaga: stara dokumentacja MediaPipe Hands mówiła odwrotnie; sprawdzone na zdjęciu testowym.)
        """
        czas = int((time.perf_counter() - self._start) * 1000)
        if czas <= self._ostatni_czas:  # MediaPipe wymaga rosnących znaczników czasu
            czas = self._ostatni_czas + 1
        self._ostatni_czas = czas

        obraz = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        wynik = self._model.detect_for_video(obraz, czas)

        dlonie = []
        for punkty, swiat, strona in zip(wynik.hand_landmarks, wynik.hand_world_landmarks, wynik.handedness):
            kategoria = strona[0]
            lewa = kategoria.category_name == "Left"
            if lustro:
                lewa = not lewa
            dlonie.append(Dlon(
                strona="Lewa" if lewa else "Prawa",
                pewnosc=float(kategoria.score),
                punkty=[(p.x, p.y, p.z) for p in punkty],
                punkty_3d=[(p.x, p.y, p.z) for p in swiat],
            ))
        return dlonie

    def zamknij(self) -> None:
        self._model.close()
