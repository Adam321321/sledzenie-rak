"""
Mierzy, ile klatek na sekundę daje sama kamera (bez śledzenia dłoni), w kilku ustawieniach.

Uruchomienie (z folderu aplikacji):
    .venv\\Scripts\\python.exe pomiary\\test_kamery.py          (kamera nr 0)
    .venv\\Scripts\\python.exe pomiary\\test_kamery.py 1        (kamera nr 1)

Jak czytać wynik:
- „stara kolejność” ~10 kl./s, „aplikacja” ~30 kl./s  -> winna była kolejność ustawień, już naprawione.
- wszędzie YUY2 i ~5–10 kl./s                         -> kamera nie przyjmuje MJPG przez DirectShow.
- MJPG, a nadal ~15 kl./s                             -> zwykle za ciemno: kamera wydłuża naświetlanie.
                                                         Zapal światło i sprawdź jeszcze raz.
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aplikacja import MJPG, _fourcc, ustaw_kamere  # noqa: E402

NUMER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEKUNDY = 4


def stara_kolejnosc(cap):  # tak, jak było w aplikacji wcześniej: MJPG przed rozdzielczością
    cap.set(cv2.CAP_PROP_FOURCC, MJPG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)


WARIANTY = {
    "stara kolejność 1080p": stara_kolejnosc,
    "aplikacja 1080p": lambda cap: ustaw_kamere(cap, [(1920, 1080)]),
    "aplikacja 720p": lambda cap: ustaw_kamere(cap, [(1280, 720)]),
}

for nazwa, ustaw in WARIANTY.items():
    cap = cv2.VideoCapture(NUMER, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Nie udało się otworzyć kamery nr {NUMER}.")
    ustaw(cap)
    for _ in range(10):  # rozgrzewka: pierwsze klatki bywają wolniejsze
        cap.read()
    klatki, start = 0, time.perf_counter()
    while time.perf_counter() - start < SEKUNDY:
        ok, _ = cap.read()
        klatki += ok
    fps = klatki / (time.perf_counter() - start)
    print(f"{nazwa:24s} {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  "
          f"format {_fourcc(cap):5s}  {fps:5.1f} kl./s")
    cap.release()
