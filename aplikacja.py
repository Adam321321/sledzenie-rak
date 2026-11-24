"""
Śledzenie dłoni — okno aplikacji.

Wybierasz kamerę (dowolną widoczną w Windowsie, także „OBS Virtual Camera”, czyli iPhone
z aplikacji Kamerka), a aplikacja rysuje na podglądzie szkielet dłoni: nadgarstek i wszystkie
stawy palców (21 punktów na dłoń, do dwóch dłoni naraz).

Biblioteki: Google MediaPipe (śledzenie), Qt / PySide6 (okno), OpenCV (obraz z kamery).
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Uruchomienie bez konsoli (pythonw): komunikaty i błędy idą do pliku zamiast w próżnię.
if sys.stderr is None or sys.stdout is None:
    _log = open(BASE / "aplikacja.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log

import cv2  # noqa: E402
from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, QThread, QTimer, Signal  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPalette, QPen  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

import dlonie  # noqa: E402
import kamery  # noqa: E402
import lokcie  # noqa: E402
import stabilizacja  # noqa: E402

# Ile modeli liczy równolegle przy dwóch dłoniach (każdy bierze co N-tą klatkę). 2 wystarcza na 30 kl./s.
# Przy jednej dłoni wystarcza jeden model, który widzi każdą klatkę, więc w ruchu śledzi dokładniej.
LICZBA_WATKOW_ANALIZY = 2

# O jaką rozdzielczość prosić kamerę. Ważne: kamera wirtualna OBS oddaje dokładnie to, o co się ją
# poprosi, i sama (bez wygładzania) przeskalowuje obraz, więc za mała prośba = pikseloza.
JAKOSCI = {"auto": [(1920, 1080), (1280, 720)], "1080": [(1920, 1080)], "720": [(1280, 720)]}
NAZWY_JAKOSCI = {"auto": "Jakość: Auto", "1080": "Jakość: 1080p", "720": "Jakość: 720p"}

TLO = QColor("#0b0d12")
KOLORY = {
    "dlon": QColor("#e8ecf3"),
    "kciuk": QColor("#ff9f0a"),
    "wskazujacy": QColor("#ffd60a"),
    "srodkowy": QColor("#30d158"),
    "serdeczny": QColor("#64d2ff"),
    "maly": QColor("#bf5af2"),
}
CZUBKI = {0, 4, 8, 12, 16, 20}
WIDOKI = [  # (nazwa w oknie, klucz zapisywany w ustawieniach)
    ("Szkielet", "szkielet"),
    ("Anatomia", "anatomia"),
    ("Anatomia + szkielet", "anatomia_oba"),
    ("Rentgen", "rentgen"),
    ("Rentgen + szkielet", "oba"),
]
WIDOKI_ZE_SZKIELETEM = {"szkielet", "anatomia_oba", "oba"}
WERSJA_WIDOKOW = 2   # 2 = jest widok Anatomia (stare zapisane „rentgen” zamienia się raz na „anatomia”)


# --------------------------------------------------------------------------- obraz + analiza (w tle)

MJPG = cv2.VideoWriter_fourcc(*"MJPG")


def _fourcc(cap) -> str:
    """Format, w jakim kamera faktycznie wysyła obraz (np. MJPG, YUY2, NV12)."""
    kod = int(cap.get(cv2.CAP_PROP_FOURCC))
    tekst = "".join(chr((kod >> 8 * i) & 0xFF) for i in range(4)).strip("\0 ")
    return tekst if tekst.isprintable() and tekst else "?"


def ustaw_kamere(cap, rozdzielczosci) -> None:
    """
    Ustawia rozdzielczość, 30 kl./s i format MJPG.

    Kolejność ma znaczenie: w DirectShow zmiana rozdzielczości (i fps) restartuje kamerę i potrafi
    przywrócić domyślny format YUY2, czyli nieskompresowany obraz. 1080p w YUY2 nie mieści się w USB 2.0
    przy 30 kl./s, więc kamera po cichu schodzi do 5–10 kl./s. Dlatego MJPG ustawiamy na końcu,
    przy docelowej rozdzielczości, i sprawdzamy, czy się przyjął.
    Kamery wirtualne (np. OBS) prośbę o MJPG ignorują i zostają przy swoim formacie, to w porządku.
    """
    for w, h in rozdzielczosci:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (w, h):
            break
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, MJPG)
    if int(cap.get(cv2.CAP_PROP_FOURCC)) != MJPG:  # część sterowników przyjmuje to dopiero za drugim razem
        cap.set(cv2.CAP_PROP_FOURCC, MJPG)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


def _klatki_na_sekunde(czasy: deque) -> float:
    return (len(czasy) - 1) / (czasy[-1] - czasy[0]) if len(czasy) > 1 and czasy[-1] > czasy[0] else 0.0


class WatekObrazu(QThread):
    """Czyta obraz z kamery (albo pliku), wysyła klatki do modeli i oddaje je oknu w dobrej kolejności."""

    klatka = Signal(QImage, object, dict)
    komunikat = Signal(str)
    blad = Signal(str)

    def __init__(self, zrodlo: int | str, nazwa: str, liczba_dloni: int, lustro: bool, wygladzanie: str,
                 jakosc: str = "auto", przedramie: bool = True):
        super().__init__()
        self.zrodlo = zrodlo          # numer kamery albo ścieżka do zdjęcia/filmu
        self.nazwa = nazwa
        self.jakosc = jakosc
        self.liczba_dloni = liczba_dloni
        self.lustro = lustro          # okno może to zmieniać w trakcie działania
        self.wygladzanie = wygladzanie  # jw.: "wyl", "slabe", "srednie", "mocne"
        self.przedramie = przedramie  # jw.: czy szukać łokci modelem ciała
        self._detektor_ramion: lokcie.DetektorRamion | None = None
        self._ramiona_blad = False
        self.format_kamery = ""       # np. "MJPG" / "YUY2"; pusty dla pliku

    def run(self) -> None:
        try:
            self._dzialaj()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.blad.emit(f"Nieoczekiwany błąd: {e}")

    def _otworz(self):
        if isinstance(self.zrodlo, int):
            cap = cv2.VideoCapture(self.zrodlo, cv2.CAP_DSHOW)
            if cap.isOpened():
                ustaw_kamere(cap, JAKOSCI.get(self.jakosc, JAKOSCI["auto"]))
                self.format_kamery = _fourcc(cap)
                print(f"[kamera] „{self.nazwa}”: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, format {self.format_kamery}, "
                      f"deklarowane {cap.get(cv2.CAP_PROP_FPS):.0f} kl./s", flush=True)
            return cap, None
        zdjecie = cv2.imread(self.zrodlo)
        if zdjecie is not None:
            return None, zdjecie
        return cv2.VideoCapture(self.zrodlo), None

    @staticmethod
    def _analizuj(detektor: dlonie.DetektorDloni, rgb, lustro: bool):
        t = time.perf_counter()
        wynik = detektor.analizuj(rgb, lustro)
        return wynik, (time.perf_counter() - t) * 1000

    def _analizuj_ramiona(self, rgb, t: float):
        """W osobnym wątku: model ciała (łokcie). Tworzony przy pierwszym użyciu, w tym samym wątku."""
        try:
            if self._detektor_ramion is None:
                self._detektor_ramion = lokcie.DetektorRamion()
            start = time.perf_counter()
            ramiona = self._detektor_ramion.analizuj(rgb, t)
            return ramiona, t, (time.perf_counter() - start) * 1000
        except Exception:  # noqa: BLE001  (bez łokci aplikacja działa dalej jak wcześniej)
            traceback.print_exc()
            self._ramiona_blad = True
            return [], t, 0.0

    def _dzialaj(self) -> None:
        if not dlonie.MODEL.exists():
            self.komunikat.emit("Pobieram model dłoni (7,8 MB, tylko za pierwszym razem)…")
            dlonie.pobierz_model()
        if self.przedramie and not lokcie.MODEL.exists():
            self.komunikat.emit("Pobieram model ciała do śledzenia łokci (9,4 MB, tylko za pierwszym razem)…")
            try:
                lokcie.pobierz_model()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                self._ramiona_blad = True

        self.komunikat.emit(f"Uruchamiam „{self.nazwa}”…")
        cap, zdjecie = self._otworz()
        if zdjecie is None and (cap is None or not cap.isOpened()):
            self.blad.emit(f"Nie udało się otworzyć „{self.nazwa}”.\n"
                           "Kamera może być wyłączona albo używa jej inna aplikacja.")
            return

        ile_modeli = 1 if self.liczba_dloni == 1 else LICZBA_WATKOW_ANALIZY
        detektory = [dlonie.DetektorDloni(liczba_dloni=self.liczba_dloni) for _ in range(ile_modeli)]
        stabilizator = stabilizacja.StabilizatorDloni(self.wygladzanie)
        wykonawcy = [ThreadPoolExecutor(max_workers=1) for _ in detektory]
        oczekujace: deque = deque()
        wejscie, wyjscie, czasy_analizy = deque(maxlen=60), deque(maxlen=60), deque(maxlen=30)
        numer = nieudane = 0
        # Model ciała (łokcie) jest wolniejszy niż model dłoni, więc liczy we własnym wątku, w swoim tempie:
        # dostaje nową klatkę, gdy skończy poprzednią, a do klatek z dłońmi dokładamy jego najnowszy wynik.
        wykonawca_ramion = ThreadPoolExecutor(max_workers=1)
        przyszlosc_ramion = None
        ramiona, t_ramion, czasy_ramion = None, -1.0, deque(maxlen=10)
        sledzenie_lokci = lokcie.SledzenieLokci()
        ostatni_raport = time.perf_counter()     # co 2 s liczby o łokciach do aplikacja.log (diagnostyka)
        try:
            while not self.isInterruptionRequested():
                if zdjecie is not None:           # zdjęcie udaje kamerę 30 kl./s
                    time.sleep(1 / 30)
                    ok, bgr = True, zdjecie.copy()
                else:
                    ok, bgr = cap.read()
                if not ok or bgr is None:
                    if isinstance(self.zrodlo, str):  # koniec filmu: od początku
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    nieudane += 1
                    if nieudane > 50:
                        self.blad.emit(f"„{self.nazwa}” przestała wysyłać obraz.")
                        return
                    continue
                nieudane = 0

                teraz = time.perf_counter()
                wejscie.append(teraz)
                lustro = self.lustro
                if lustro:
                    bgr = cv2.flip(bgr, 1)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                i = numer % len(detektory)
                oczekujace.append((rgb, teraz, wykonawcy[i].submit(self._analizuj, detektory[i], rgb, lustro)))
                numer += 1

                przedramie = self.przedramie and not self._ramiona_blad
                if przyszlosc_ramion is not None and przyszlosc_ramion.done():
                    ramiona, t_ramion, ms_ramion = przyszlosc_ramion.result()
                    czasy_ramion.append(ms_ramion)
                    przyszlosc_ramion = None
                if przedramie and przyszlosc_ramion is None:
                    przyszlosc_ramion = wykonawca_ramion.submit(self._analizuj_ramiona, rgb, teraz)
                if not przedramie:
                    ramiona = None

                # Oddaj gotowe klatki w kolejności; jeśli wszystkie modele są zajęte, poczekaj na najstarszą.
                while oczekujace and (oczekujace[0][2].done() or len(oczekujace) > len(detektory)):
                    rgb0, t0, przyszlosc = oczekujace.popleft()
                    wynik, ms = przyszlosc.result()
                    czasy_analizy.append(ms)
                    h, w = rgb0.shape[:2]
                    stabilizator.poziom = self.wygladzanie
                    wynik = stabilizator.aktualizuj(wynik, t0, w, h)  # klatki idą tu po kolei
                    if przedramie:
                        sledzenie_lokci.przypisz(wynik, ramiona, t_ramion, t0, w, h)
                    koniec = time.perf_counter()
                    if przedramie and koniec - ostatni_raport >= 2.0:
                        ms = sum(czasy_ramion) / len(czasy_ramion) if czasy_ramion else 0.0
                        print(f"[łokieć {time.strftime('%H:%M:%S')}] {w}×{h}, model ciała {ms:.0f} ms, "
                              f"{koniec - ostatni_raport:.1f} s: {sledzenie_lokci.raport()}", flush=True)
                        ostatni_raport = koniec
                    wyjscie.append(koniec)
                    obraz = QImage(rgb0.data, w, h, rgb0.strides[0], QImage.Format.Format_RGB888).copy()
                    self.klatka.emit(obraz, wynik, {
                        "kamera_fps": _klatki_na_sekunde(wejscie),
                        "format": self.format_kamery,
                        "fps": _klatki_na_sekunde(wyjscie),
                        "analiza_ms": sum(czasy_analizy) / len(czasy_analizy),
                        "opoznienie_ms": (koniec - t0) * 1000,
                        "rozmiar": (w, h),
                        "lustro": lustro,
                        "przedramie": przedramie,
                        "ramiona_ms": sum(czasy_ramion) / len(czasy_ramion) if czasy_ramion else 0.0,
                    })
        finally:
            for wyk in (*wykonawcy, wykonawca_ramion):
                wyk.shutdown(wait=True)
            for d in detektory:
                d.zamknij()
            if self._detektor_ramion is not None:
                self._detektor_ramion.zamknij()
            if cap is not None:
                cap.release()


# --------------------------------------------------------------------------- podgląd ze szkieletem

class RysowanieDloni:
    """Wspólne dla obu podglądów: stan (obraz, dłonie, komunikat) i rysowanie szkieletu 2D na wierzchu."""

    def _przygotuj_stan(self) -> None:
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._obraz: QImage | None = None
        self._dlonie: list[dlonie.Dlon] = []
        self._komunikat = "Wybierz kamerę i kliknij Start"
        self._komunikat_blad = False
        self._nowy_obraz = False
        self.pokaz_numery = False
        self.lustro = False
        self.widok = "szkielet"            # klucz z WIDOKI
        self.warstwy = {"kosci"}           # warstwy anatomii/rentgena: "kosci", "miesnie", "sciegna"

    def pokaz(self, obraz: QImage, dlonie_: list, lustro: bool = False) -> None:
        self._obraz, self._dlonie, self.lustro = obraz, dlonie_, lustro
        self._nowy_obraz = True
        self.update()

    def komunikat(self, tekst: str, blad: bool = False) -> None:
        self._obraz, self._dlonie = None, []
        self._komunikat, self._komunikat_blad = tekst, blad
        self.update()

    def _prostokat_obrazu(self) -> QRectF:
        w, h = self._obraz.width(), self._obraz.height()
        s = min(self.width() / w, self.height() / h)
        return QRectF((self.width() - w * s) / 2, (self.height() - h * s) / 2, w * s, h * s)

    def _rysuj_komunikat(self, p: QPainter) -> None:
        p.fillRect(self.rect(), TLO)
        p.setPen(QColor("#ff6b61") if self._komunikat_blad else QColor("#9aa3b2"))
        f = p.font()
        f.setPointSizeF(12.5)
        p.setFont(f)
        p.drawText(self.rect().adjusted(32, 32, -32, -32),
                   Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._komunikat)

    def _rysuj_nakladke(self, p: QPainter, cel: QRectF) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for d in self._dlonie:
            p.setOpacity(d.widocznosc)  # chwilowo zgubiona dłoń przygasa, zamiast mignąć
            try:
                g, pkt = self._geometria_dloni(d, cel)
                if self.widok in WIDOKI_ZE_SZKIELETEM:
                    if d.lokiec is not None:
                        self._rysuj_przedramie(p, pkt[0], QPointF(cel.x() + d.lokiec[0] * cel.width(),
                                                                  cel.y() + d.lokiec[1] * cel.height()), g, cel)
                    self._rysuj_szkielet(p, pkt, g)
                self._rysuj_podpis(p, d, pkt, g)
            finally:
                p.setOpacity(1.0)

    @staticmethod
    def _geometria_dloni(d: dlonie.Dlon, cel: QRectF) -> tuple[float, list[QPointF]]:
        g = max(2.0, cel.height() / 220)  # grubość kości zależna od wielkości podglądu
        return g, [QPointF(cel.x() + x * cel.width(), cel.y() + y * cel.height()) for x, y, _ in d.punkty]

    @staticmethod
    def _rysuj_przedramie(p: QPainter, nadgarstek: QPointF, lokiec: QPointF, g: float, cel: QRectF) -> None:
        """Przedramię: od nadgarstka do łokcia (łokieć bywa poza kadrem, więc rysujemy tylko w obrazie)."""
        p.save()
        p.setClipRect(cel)
        p.setPen(QPen(QColor(0, 0, 0, 140), g + 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(nadgarstek, lokiec)
        p.setPen(QPen(KOLORY["dlon"], g, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(nadgarstek, lokiec)
        p.setPen(QPen(QColor(0, 0, 0, 170), 1.5))
        p.setBrush(KOLORY["dlon"])
        p.drawEllipse(lokiec, g * 2.6, g * 2.6)
        p.restore()

    def _rysuj_szkielet(self, p: QPainter, pkt: list[QPointF], g: float) -> None:

        # Najpierw ciemny cień pod wszystkimi kośćmi, potem kolorowe kości: dobrze widać na każdym tle.
        p.setPen(QPen(QColor(0, 0, 0, 140), g + 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for kosci in dlonie.KOSCI.values():
            for a, b in kosci:
                p.drawLine(pkt[a], pkt[b])
        for grupa, kosci in dlonie.KOSCI.items():
            p.setPen(QPen(KOLORY[grupa], g, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for a, b in kosci:
                p.drawLine(pkt[a], pkt[b])

        p.setPen(QPen(QColor(0, 0, 0, 170), 1.5))
        for i, q in enumerate(pkt):
            r = g * (2.3 if i in CZUBKI else 1.6)
            p.setBrush(KOLORY[dlonie.palec_punktu(i)])
            p.drawEllipse(q, r, r)

        if self.pokaz_numery:
            f = QFont(self.font())
            f.setPointSizeF(8.5)
            f.setBold(True)
            p.setFont(f)
            for i, q in enumerate(pkt):
                tekst_rect = QRectF(q.x() + g * 2, q.y() - g * 2 - 14, 28, 14)
                p.setPen(QColor(0, 0, 0, 200))
                p.drawText(tekst_rect.translated(1, 1), Qt.AlignmentFlag.AlignLeft, str(i))
                p.setPen(QColor("#ffffff"))
                p.drawText(tekst_rect, Qt.AlignmentFlag.AlignLeft, str(i))

    def _rysuj_podpis(self, p: QPainter, d: dlonie.Dlon, pkt: list[QPointF], g: float) -> None:
        """Podpis przy nadgarstku: która dłoń i jak pewny jest model."""
        f = QFont(self.font())
        f.setPointSizeF(10)
        f.setBold(True)
        p.setFont(f)
        tekst = f"{d.strona}  {d.pewnosc:.0%}"
        m = QFontMetricsF(f)
        prost = QRectF(0, 0, m.horizontalAdvance(tekst) + 16, m.height() + 8)
        prost.moveCenter(QPointF(pkt[0].x(), pkt[0].y() + g * 4 + prost.height() / 2 + 4))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 175))
        p.drawRoundedRect(prost, 6, 6)
        p.setPen(QColor("#64d2ff") if d.strona == "Lewa" else QColor("#ff9f0a"))
        p.drawText(prost, Qt.AlignmentFlag.AlignCenter, tekst)


class Podglad(RysowanieDloni, QWidget):
    """
    Podgląd: obraz z kamery, w trybie rentgena przepuszczony przez rentgen 3D (karta graficzna,
    własny kontekst OpenGL), a na wierzchu szkielet 2D i podpisy (QPainter).
    """

    rentgen_niedostepny = Signal(str)

    def __init__(self):
        QWidget.__init__(self)
        self._przygotuj_stan()
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._rtg = None              # rentgen.RentgenPozaEkranem, tworzony przy pierwszym użyciu
        self._rtg_blad = False
        self._rtg_klatka: QImage | None = None

    def _rentgen(self):
        if self._rtg is None and not self._rtg_blad:
            try:
                import rentgen
                self._rtg = rentgen.RentgenPozaEkranem()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._rtg_blad = True
                self.rentgen_niedostepny.emit(str(e))
        return self._rtg

    def paintEvent(self, _event) -> None:  # noqa: N802 (nazwa wymagana przez Qt)
        p = QPainter(self)
        if self._obraz is None:
            self._rysuj_komunikat(p)
            return
        p.fillRect(self.rect(), TLO)
        cel = self._prostokat_obrazu()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rtg = self._rentgen() if self.widok != "szkielet" else None
        if rtg is not None:
            p.drawImage(cel, self._klatka_rentgena(rtg, cel))
        else:
            p.drawImage(cel, self._obraz)
        self._rysuj_nakladke(p, cel)

    def _klatka_rentgena(self, rtg, cel: QRectF) -> QImage:
        w, h = self._obraz.width(), self._obraz.height()
        dpr = self.devicePixelRatioF()
        # Rysujemy w rozmiarze podglądu na ekranie (ostro), ale nie większym niż obraz z kamery.
        skala = min(1.0, cel.width() * dpr / w)
        cs, ch = max(2, round(w * skala)), max(2, round(h * skala))
        styl = "anatomia" if self.widok.startswith("anatomia") else "rentgen"
        tablica = rtg.klatka(self._obraz.constBits(), w, h, 4, self._nowy_obraz, self._dlonie, self.lustro,
                             self.warstwy, cs, ch, styl)
        self._nowy_obraz = False
        tablica = np.ascontiguousarray(tablica)
        self._rtg_klatka = QImage(tablica.data, cs, ch, tablica.strides[0], QImage.Format.Format_RGB888).copy()
        return self._rtg_klatka


# --------------------------------------------------------------------------- okno

class OknoGlowne(QMainWindow):
    def __init__(self, plik: str | None = None):
        super().__init__()
        self.setWindowTitle("Śledzenie dłoni")
        self.ustawienia = QSettings("RBStudio", "SledzenieDloni")
        self.watek: WatekObrazu | None = None
        self._plik = plik
        self.zapisuj_ustawienia = True

        self.kamera = QComboBox()
        self.kamera.setMinimumWidth(280)
        self.odswiez = QToolButton()
        self.odswiez.setText("↻")
        self.odswiez.setToolTip("Odśwież listę kamer")
        self.przycisk = QPushButton("Start")
        self.przycisk.setObjectName("start")
        self.przycisk.setMinimumWidth(110)
        self.liczba_dloni = QComboBox()
        self.liczba_dloni.addItem("1 dłoń", 1)
        self.liczba_dloni.addItem("2 dłonie", 2)
        self.liczba_dloni.setToolTip("Jedna dłoń liczy się dwa razy szybciej i dokładniej śledzi szybki ruch")
        self.jakosc = QComboBox()
        for klucz, nazwa in NAZWY_JAKOSCI.items():
            self.jakosc.addItem(nazwa, klucz)
        self.jakosc.setToolTip("Rozdzielczość obrazu z kamery. Auto = najwyższa, jaką kamera obsługuje (do 1080p).")
        self.wygladzanie = QComboBox()
        for klucz, nazwa in stabilizacja.NAZWY_POZIOMOW.items():
            self.wygladzanie.addItem(f"Wygładzanie: {nazwa}", klucz)
        self.wygladzanie.setToolTip(
            "Tłumi drżenie punktów. Mocniejsze = spokojniej, ale przy bardzo szybkim ruchu szkielet\n"
            "minimalnie się spóźnia. „Wył.” pokazuje surowy wynik modelu (do porównania).")
        self.lustro = QCheckBox("Lustro")
        self.lustro.setToolTip("Odbicie lustrzane, jak w selfie: ruszasz prawą ręką, ruszy się ta po prawej")
        self.numery = QCheckBox("Numery punktów")
        self.numery.setToolTip("Pokaż numery 0–20 przy punktach (przydatne przy programowaniu gestów)")
        self.przedramie = QCheckBox("Łokieć")
        self.przedramie.setToolTip(
            "Śledzenie łokcia modelem ciała (MediaPipe Pose): przedramię w widokach Anatomia i Rentgen\n"
            "idzie od nadgarstka do łokcia, także przy zgiętym nadgarstku. Łokieć musi być w kadrze,\n"
            "a model ciała szuka najpierw głowy i tułowia. Wyłączone = mniej pracy dla procesora.")
        self.widok = QComboBox()
        for nazwa, klucz in WIDOKI:
            self.widok.addItem(f"Widok: {nazwa}", klucz)
        self.widok.setToolTip(
            "Anatomia: kości, mięśnie i ścięgna jak na rycinie z atlasu, dopasowane do twojej dłoni (poglądowo).\n"
            "Rentgen: to samo jako świecąca klisza, warstwy prześwitują przez siebie.")
        self.warstwy = {
            "kosci": QCheckBox("Kości"),
            "miesnie": QCheckBox("Mięśnie"),
            "sciegna": QCheckBox("Ścięgna"),
        }

        pasek = QHBoxLayout()
        pasek.setSpacing(8)
        pasek.addWidget(QLabel("Kamera:"))
        pasek.addWidget(self.kamera)
        pasek.addWidget(self.odswiez)
        pasek.addWidget(self.przycisk)
        pasek.addSpacing(18)
        pasek.addWidget(self.jakosc)
        pasek.addWidget(self.liczba_dloni)
        pasek.addWidget(self.wygladzanie)
        pasek.addStretch(1)

        pasek2 = QHBoxLayout()
        pasek2.setSpacing(8)
        pasek2.addWidget(self.widok)
        pasek2.addSpacing(4)
        for pole in self.warstwy.values():
            pasek2.addWidget(pole)
        pasek2.addSpacing(18)
        pasek2.addWidget(self.lustro)
        pasek2.addWidget(self.numery)
        pasek2.addWidget(self.przedramie)
        pasek2.addStretch(1)

        self.podglad = Podglad()
        self.podglad.rentgen_niedostepny.connect(self._brak_rentgena)
        srodek = QWidget()
        self._uklad = QVBoxLayout(srodek)
        self._uklad.setContentsMargins(0, 0, 0, 0)
        self._uklad.setSpacing(0)
        gora = QWidget()
        gora.setObjectName("pasek")
        paski = QVBoxLayout(gora)
        paski.setContentsMargins(12, 8, 12, 8)
        paski.setSpacing(6)
        paski.addLayout(pasek)
        paski.addLayout(pasek2)
        self._uklad.addWidget(gora)
        self._uklad.addWidget(self.podglad, 1)
        self.setCentralWidget(srodek)

        self.statystyki = QLabel("Gotowe")
        self.statusBar().addWidget(self.statystyki, 1)
        self.statusBar().addPermanentWidget(QLabel(
            "Google MediaPipe · anatomia: BodyParts3D © DBCLS, CC BY 4.0  "))

        # Zapamiętane ustawienia
        self.lustro.setChecked(self.ustawienia.value("lustro", True, type=bool))
        self.numery.setChecked(self.ustawienia.value("numery", False, type=bool))
        self.przedramie.setChecked(self.ustawienia.value("przedramie", True, type=bool))
        self.liczba_dloni.setCurrentIndex(1 if self.ustawienia.value("dlonie", 2, type=int) == 2 else 0)
        indeks = self.jakosc.findData(self.ustawienia.value("jakosc", "auto", type=str))
        self.jakosc.setCurrentIndex(max(0, indeks))
        indeks = self.wygladzanie.findData(self.ustawienia.value("wygladzanie", "srednie", type=str))
        self.wygladzanie.setCurrentIndex(indeks if indeks >= 0 else self.wygladzanie.findData("srednie"))
        widok = self.ustawienia.value("widok", "szkielet", type=str)
        zapisane_warstwy = self.ustawienia.value("warstwy", "kosci", type=str).split(",")
        if self.ustawienia.value("wersja_widokow", 1, type=int) < WERSJA_WIDOKOW and widok in ("rentgen", "oba"):
            # Jednorazowo po dodaniu widoku Anatomia: zapisany stary rentgen zamienia się na nowy styl
            # ze wszystkimi warstwami (nowa wersja zapisuje się przy zamknięciu okna).
            widok = "anatomia" if widok == "rentgen" else "anatomia_oba"
            zapisane_warstwy = ["kosci", "miesnie", "sciegna"]
        self.widok.setCurrentIndex(max(0, self.widok.findData(widok)))
        for klucz, pole in self.warstwy.items():
            pole.setChecked(klucz in zapisane_warstwy)
        self.podglad.pokaz_numery = self.numery.isChecked()
        self._zastosuj_widok()
        geometria = self.ustawienia.value("geometria")
        if geometria is not None:
            self.restoreGeometry(geometria)
        else:
            self.resize(1200, 780)

        self.wypelnij_liste_kamer()

        self.przycisk.clicked.connect(self.przelacz)
        self.odswiez.clicked.connect(self.wypelnij_liste_kamer)
        self.kamera.activated.connect(self._zmiana_zrodla)
        self.liczba_dloni.activated.connect(self._zmiana_zrodla)
        self.jakosc.activated.connect(self._zmiana_zrodla)
        self.lustro.toggled.connect(self._zmiana_lustra)
        self.wygladzanie.currentIndexChanged.connect(self._zmiana_wygladzania)
        self.numery.toggled.connect(self._zmiana_numerow)
        self.przedramie.toggled.connect(self._zmiana_przedramienia)
        self.widok.currentIndexChanged.connect(self._zastosuj_widok)
        for pole in self.warstwy.values():
            pole.toggled.connect(self._zastosuj_widok)

    # ---- widok i rentgen
    def _zastosuj_widok(self, *_args) -> None:
        self.podglad.widok = self.widok.currentData()
        self.podglad.warstwy = {k for k, pole in self.warstwy.items() if pole.isChecked()}
        for pole in self.warstwy.values():
            pole.setEnabled(self.widok.isEnabled() and self.podglad.widok != "szkielet")
        self.podglad.update()

    def _brak_rentgena(self, opis: str) -> None:
        """Karta graficzna/OpenGL nie zadziałały: zostaje zwykły szkielet 2D zamiast wysypania aplikacji."""
        self.widok.setCurrentIndex(0)
        self.widok.setEnabled(False)
        self.widok.setToolTip(f"Rentgen niedostępny: OpenGL nie działa ({opis})")
        self._zastosuj_widok()
        self.statystyki.setText("Rentgen niedostępny na tym komputerze (OpenGL). Działa zwykły szkielet.")

    # ---- kamery
    def wypelnij_liste_kamer(self) -> None:
        poprzednia = self.kamera.currentText() or self.ustawienia.value("kamera", "", type=str)
        self.kamera.clear()
        try:
            for i, nazwa in enumerate(kamery.lista_kamer()):
                self.kamera.addItem(nazwa, i)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.podglad.komunikat(f"Nie udało się pobrać listy kamer: {e}", blad=True)
        if self._plik:
            self.kamera.addItem(f"Plik: {Path(self._plik).name}", self._plik)
        if self.kamera.count() == 0:
            self.kamera.addItem("(brak kamer)", None)
        indeks = self.kamera.findText(poprzednia)
        if indeks >= 0:
            self.kamera.setCurrentIndex(indeks)

    # ---- start / stop
    def przelacz(self) -> None:
        if self.watek is not None:
            self.zatrzymaj()
            self.podglad.komunikat("Zatrzymano. Kliknij Start, żeby wznowić.")
        else:
            self.uruchom()

    def uruchom(self) -> None:
        zrodlo = self.kamera.currentData()
        if zrodlo is None:
            self.podglad.komunikat("Nie znaleziono żadnej kamery. Podłącz kamerę i kliknij ↻.", blad=True)
            return
        self.watek = WatekObrazu(zrodlo, self.kamera.currentText(), self.liczba_dloni.currentData(),
                                 self.lustro.isChecked(), self.wygladzanie.currentData(), self.jakosc.currentData(),
                                 self.przedramie.isChecked())
        self.watek.klatka.connect(self._nowa_klatka)
        self.watek.komunikat.connect(lambda tekst: self.podglad.komunikat(tekst))
        self.watek.blad.connect(self._blad)
        self.watek.finished.connect(self._watek_skonczyl)
        self.watek.start()
        self._ustaw_przycisk(True)
        self.statystyki.setText("Uruchamiam…")

    def zatrzymaj(self) -> None:
        watek, self.watek = self.watek, None
        if watek is not None:
            watek.klatka.disconnect()
            watek.requestInterruption()
            watek.wait(5000)
        self._ustaw_przycisk(False)
        self.statystyki.setText("Zatrzymano")

    def _watek_skonczyl(self) -> None:
        if self.sender() is self.watek:  # wątek sam się zakończył (np. błąd kamery)
            self.watek = None
            self._ustaw_przycisk(False)

    def _ustaw_przycisk(self, dziala: bool) -> None:
        self.przycisk.setText("Stop" if dziala else "Start")
        self.przycisk.setProperty("dziala", dziala)
        self.przycisk.style().unpolish(self.przycisk)
        self.przycisk.style().polish(self.przycisk)

    def _zmiana_zrodla(self, *_args) -> None:
        if self.watek is not None:  # zmiana kamery albo liczby dłoni w trakcie: uruchom od nowa
            self.zatrzymaj()
            self.uruchom()

    def _zmiana_lustra(self, wl: bool) -> None:
        if self.watek is not None:
            self.watek.lustro = wl

    def _zmiana_wygladzania(self, *_args) -> None:
        if self.watek is not None:  # zmienia się w locie, bez restartu kamery
            self.watek.wygladzanie = self.wygladzanie.currentData()

    def _zmiana_numerow(self, wl: bool) -> None:
        self.podglad.pokaz_numery = wl
        self.podglad.update()

    def _zmiana_przedramienia(self, wl: bool) -> None:
        if self.watek is not None:  # włącza się w locie, bez restartu kamery
            self.watek.przedramie = wl

    # ---- dane z wątku
    def _nowa_klatka(self, obraz: QImage, dlonie_: list, s: dict) -> None:
        self.podglad.pokaz(obraz, dlonie_, s["lustro"])
        strony = ", ".join(d.strona for d in dlonie_)
        dlonie_txt = f"Dłonie: {len(dlonie_)}" + (f" ({strony})" if strony else "")
        if s.get("przedramie"):
            lokcie_ = sum(d.lokiec is not None for d in dlonie_)
            dlonie_txt += f"     ·     łokcie: {lokcie_}"
            if s.get("ramiona_ms"):
                dlonie_txt += f" (model ciała {s['ramiona_ms']:.0f} ms)"
        w, h = s["rozmiar"]
        # „kamera” = ile klatek daje sama kamera, „obraz” = ile przechodzi przez analizę na ekran.
        # Jeśli niska jest już kamera, hamulec jest w kamerze/USB/oświetleniu, nie w analizie.
        kamera_txt = f"kamera {s['kamera_fps']:.0f} kl./s" + (f" {s['format']}" if s.get("format") else "")
        self.statystyki.setText(
            f"{dlonie_txt}     ·     obraz {s['fps']:.0f} kl./s ({kamera_txt})"
            f"     ·     analiza {s['analiza_ms']:.0f} ms"
            f"     ·     opóźnienie {s['opoznienie_ms']:.0f} ms     ·     {w}×{h}")

    def _blad(self, tekst: str) -> None:
        self.podglad.komunikat(tekst, blad=True)
        self.statystyki.setText("Błąd")

    # ---- zamykanie
    def closeEvent(self, event) -> None:  # noqa: N802
        self.zatrzymaj()
        if not self.zapisuj_ustawienia:
            super().closeEvent(event)
            return
        self.ustawienia.setValue("geometria", self.saveGeometry())
        self.ustawienia.setValue("kamera", self.kamera.currentText())
        self.ustawienia.setValue("lustro", self.lustro.isChecked())
        self.ustawienia.setValue("numery", self.numery.isChecked())
        self.ustawienia.setValue("przedramie", self.przedramie.isChecked())
        self.ustawienia.setValue("dlonie", self.liczba_dloni.currentData())
        self.ustawienia.setValue("wygladzanie", self.wygladzanie.currentData())
        self.ustawienia.setValue("jakosc", self.jakosc.currentData())
        if self.widok.isEnabled():
            self.ustawienia.setValue("widok", self.widok.currentData())
        self.ustawienia.setValue("warstwy", ",".join(k for k, p in self.warstwy.items() if p.isChecked()))
        self.ustawienia.setValue("wersja_widokow", WERSJA_WIDOKOW)
        super().closeEvent(event)


# --------------------------------------------------------------------------- wygląd i start

def ciemny_motyw(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    kolory = {
        QPalette.ColorRole.Window: "#14171e", QPalette.ColorRole.WindowText: "#e6e9ef",
        QPalette.ColorRole.Base: "#0f1217", QPalette.ColorRole.AlternateBase: "#1a1e26",
        QPalette.ColorRole.Text: "#e6e9ef", QPalette.ColorRole.Button: "#1f2430",
        QPalette.ColorRole.ButtonText: "#e6e9ef", QPalette.ColorRole.Highlight: "#0a84ff",
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: "#1f2430",
        QPalette.ColorRole.ToolTipText: "#e6e9ef", QPalette.ColorRole.PlaceholderText: "#7d8595",
    }
    for rola, kolor in kolory.items():
        pal.setColor(rola, QColor(kolor))
    app.setPalette(pal)
    app.setStyleSheet("""
        QWidget#pasek { background: #14171e; border-bottom: 1px solid #262b36; }
        QPushButton#start { background: #0a84ff; color: white; border: none; border-radius: 6px;
                            padding: 7px 18px; font-weight: 600; }
        QPushButton#start:hover { background: #2b95ff; }
        QPushButton#start[dziala="true"] { background: #ff453a; }
        QPushButton#start[dziala="true"]:hover { background: #ff6259; }
        QComboBox, QToolButton { padding: 5px 8px; }
        QStatusBar { background: #14171e; color: #9aa3b2; border-top: 1px solid #262b36; }
        QStatusBar QLabel { color: #9aa3b2; padding: 2px 6px; }
    """)


def pokaz_blad(typ, wartosc, slad) -> None:
    tekst = "".join(traceback.format_exception(typ, wartosc, slad))
    print(tekst, file=sys.stderr)
    if QApplication.instance() is not None:
        QMessageBox.critical(None, "Śledzenie dłoni – błąd", tekst[-3000:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Śledzenie dłoni z kamery")
    parser.add_argument("--plik", help="zdjęcie albo film zamiast kamery (np. do testów)")
    parser.add_argument("--start", action="store_true", help="od razu kliknij Start")
    parser.add_argument("--zrzut", help="(testy) po kilku sekundach zapisz zrzut okna do PNG i zamknij")
    parser.add_argument("--po", type=float, default=4.0, help="(testy) po ilu sekundach zrobić zrzut")
    parser.add_argument("--widok", choices=[k for _n, k in WIDOKI], help="(testy) ustaw widok na starcie")
    parser.add_argument("--warstwy", help="(testy) np. kosci,miesnie,sciegna")
    parser.add_argument("--poza-ekranem", action="store_true", help="(testy) okno poza widocznym ekranem")
    parser.add_argument("--bez-lokcia", action="store_true", help="(testy) wyłącz śledzenie łokci")
    args = parser.parse_args()

    sys.excepthook = pokaz_blad
    app = QApplication(sys.argv)
    app.setApplicationName("Śledzenie dłoni")
    ciemny_motyw(app)
    okno = OknoGlowne(plik=args.plik)
    if args.plik:
        okno.kamera.setCurrentIndex(okno.kamera.count() - 1)
    if args.widok:
        okno.widok.setCurrentIndex(okno.widok.findData(args.widok))
    if args.warstwy:
        for klucz, pole in okno.warstwy.items():
            pole.setChecked(klucz in args.warstwy.split(","))
    if args.bez_lokcia:
        okno.przedramie.setChecked(False)
    if args.poza_ekranem:
        okno.move(-5000, -5000)
    okno.show()
    if args.start or args.plik:
        QTimer.singleShot(0, okno.uruchom)
    if args.zrzut:
        okno.zapisuj_ustawienia = False

        def zrzut():
            okno.grab().save(args.zrzut)
            print("STATYSTYKI:", okno.statystyki.text())
            okno.close()
        QTimer.singleShot(int(args.po * 1000), zrzut)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
