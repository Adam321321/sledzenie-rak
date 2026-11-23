"""
Pomiar stabilności: drgania w bezruchu i odstawanie (opóźnienie) w ruchu, surowo vs. z wygładzaniem.
Użycie: python pomiar_stabilnosci.py <zdjęcie> [plik_cache.npz]
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"A:\sledzenie-dloni")
import cv2  # noqa: E402

import dlonie  # noqa: E402
import stabilizacja  # noqa: E402

FPS, N, A, OKRES = 30, 150, 160, 1.5
foto = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
H, W = foto.shape[:2]
SZER = W + 2 * A


LOS = np.random.default_rng(1)


def klatka(off: float) -> np.ndarray:
    """Klatka jak z prawdziwej kamerki: szum matrycy, lekkie migotanie jasności i kompresja JPEG."""
    c = np.zeros((H, SZER, 3), np.float32)
    x = A + int(round(off))
    c[:, x:x + W] = foto
    c = c * LOS.uniform(0.97, 1.03) + LOS.normal(0, 4.0, c.shape)
    c = np.clip(c, 0, 255).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.imdecode(jpg, cv2.IMREAD_UNCHANGED)


def przesuniecie(i: int, ruch: bool) -> float:
    return A * np.sin(2 * np.pi * (i / FPS) / OKRES) if ruch else 0.0


MODELE = int(__import__("os").environ.get("LICZBA_MODELI", "2"))


def surowe(ruch: bool):
    dets = [dlonie.DetektorDloni() for _ in range(MODELE)]  # jak w aplikacji: na zmianę
    wyniki = [dets[i % MODELE].analizuj(klatka(przesuniecie(i, ruch)), lustro=False) for i in range(N)]
    for d in dets:
        d.zamknij()
    return wyniki


cache = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if cache and cache.exists():
    dane = pickle.loads(cache.read_bytes())
else:
    dane = {"bezruch": surowe(False), "ruch": surowe(True)}
    if cache:
        cache.write_bytes(pickle.dumps(dane))


def po_pozycji(dl):
    """Rozdziela dłonie na górną i dolną (po wysokości nadgarstka) -> {nazwa: (punkty px 21x2, strona)}."""
    out = {}
    for d in dl:
        p = np.asarray(d.punkty)[:, :2] * [SZER, H]
        out["gorna" if p[0, 1] < 0.6 * H else "dolna"] = (p, d.strona)
    return out


def ocen(poziom: str):
    wynik = {}
    for scen in ("bezruch", "ruch"):
        st = stabilizacja.StabilizatorDloni(poziom)
        seria = [po_pozycji(st.aktualizuj(dl, i / FPS, SZER, H)) for i, dl in enumerate(dane[scen])]
        wynik[scen] = seria
    return wynik


# Prawdziwe położenie: średnia surowych punktów w bezruchu (szum się uśrednia).
sur_bez = [po_pozycji(dl) for dl in dane["bezruch"]]
prawda = {k: np.mean([f[k][0] for f in sur_bez[10:] if k in f], axis=0) for k in ("gorna", "dolna")}

print(f"{'poziom':<10}{'drganie w bezruchu':>22}{'odstawanie w ruchu':>22}{'zmiany podpisu':>17}{'brakujące klatki':>18}")
for poziom in ("wyl", "slabe", "srednie", "mocne"):
    w = ocen(poziom)
    # drganie: średni ruch punktu między kolejnymi klatkami, gdy nic się nie rusza
    ruchy = [np.linalg.norm(a[k][0] - b[k][0], axis=1).mean()
             for a, b in zip(w["bezruch"][10:], w["bezruch"][11:]) for k in prawda if k in a and k in b]
    # odstawanie: średnia odległość od prawdziwego położenia w ruchu
    bledy = [np.linalg.norm(f[k][0] - (prawda[k] + [przesuniecie(i, True), 0]), axis=1).mean()
             for i, f in enumerate(w["ruch"]) if i >= 10 for k in prawda if k in f]
    zmiany = sum(1 for scen in w.values() for a, b in zip(scen[10:], scen[11:])
                 for k in prawda if k in a and k in b and a[k][1] != b[k][1])
    braki = sum(1 for scen in w.values() for f in scen[10:] for k in prawda if k not in f)
    print(f"{stabilizacja.NAZWY_POZIOMOW[poziom]:<10}{np.mean(ruchy):>19.2f} px{np.mean(bledy):>19.2f} px"
          f"{zmiany:>17}{braki:>18}")
