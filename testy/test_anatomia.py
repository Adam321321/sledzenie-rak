"""
Widok „Anatomia” bez kamery:
1) wzorcowa dłoń ustawiona wnętrzem i grzbietem do kamery (do porównania z atlasem) -> anatomia_wzorzec.png
2) przedramię: bez łokcia / łokieć w linii / nadgarstek zgięty w bok o 35° / grzbiet -> anatomia_przedramie.png
3) model nałożony na zdjęcie testowe (powiększone ×3, wycinek dolnej dłoni) -> anatomia_zdjecie.png
"""
import sys
import time
from pathlib import Path

import cv2
import moderngl
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import dlonie  # noqa: E402
import rentgen  # noqa: E402
import stabilizacja  # noqa: E402

TESTY = Path(__file__).resolve().parent
WSZYSTKO = {"kosci", "miesnie", "sciegna"}


def dlon_wzorcowa(R: np.ndarray, E: np.ndarray, wnetrze: bool, W: int, H: int, px_na_mm: float,
                  lokiec_kat: float | None = None, srodek_y: float = 0.5) -> dlonie.Dlon:
    """
    Wzorcowa (prawa) dłoń, palce do góry, wnętrzem albo grzbietem do kamery.
    lokiec_kat: None = bez łokcia; inaczej łokieć wzorcowy obrócony o tyle stopni w płaszczyźnie obrazu
    wokół nadgarstka (0 = przedramię w linii z dłonią, inne = nadgarstek zgięty w bok).
    """
    u = R[9] - R[0]
    u /= np.linalg.norm(u)
    n = np.cross(R[5] - R[0], R[17] - R[0])      # dla prawej dłoni: strona dłoniowa
    n -= np.dot(n, u) * u
    n /= np.linalg.norm(n)
    zrodlo = np.stack([u, n, np.cross(u, n)], axis=1)
    cel_n = (0, 0, -1) if wnetrze else (0, 0, 1)   # -z = do kamery
    cel_u = np.array([0.0, -1.0, 0.0])           # palce do góry obrazu
    cel = np.stack([cel_u, cel_n, np.cross(cel_u, cel_n)], axis=1)
    obrot = cel @ zrodlo.T
    P_mm = (obrot @ (R - R[0]).T).T
    przesuniecie = np.array([W / 2, H * srodek_y]) - P_mm[[0, 9], :2].mean(axis=0) * px_na_mm
    px = P_mm[:, :2] * px_na_mm + przesuniecie
    d = dlonie.Dlon("Prawa", 1.0, [(x / W, y / H, 0.0) for x, y in px], [tuple(p / 1000.0) for p in P_mm])
    if lokiec_kat is not None:
        e = obrot @ (E - R[0])
        a = np.radians(lokiec_kat)
        e[:2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] @ e[:2]
        ex, ey = e[:2] * px_na_mm + przesuniecie
        d.lokiec = (ex / W, ey / H, e[2] / 1000.0)
    return d


def klatka(r: rentgen.RenderRentgen, fbo, W, H, dl, warstwy, styl="anatomia"):
    fbo.use()
    fbo.clear(0.08, 0.09, 0.11, 1.0)
    r.rysuj(fbo, (0, 0, W, H), W, H, dl, lustro=False, warstwy=warstwy, przyciemnienie=0.25, styl=styl)
    return np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)[::-1]


ctx = moderngl.create_standalone_context(require=330)
r = rentgen.RenderRentgen(ctx)
R, E = r.model.punkty, r.model.lokiec

# 1) Wzorzec (sama dłoń, jak dotąd)
W, H = 700, 1000
fbo = ctx.simple_framebuffer((W, H), components=4)
kadry = []
for wnetrze, warstwy in ((True, {"kosci"}), (True, {"kosci", "miesnie"}), (True, WSZYSTKO), (False, WSZYSTKO)):
    kadry.append(klatka(r, fbo, W, H, [dlon_wzorcowa(R, E, wnetrze, W, H, 4.0, srodek_y=0.56)], warstwy))
cv2.imwrite(str(TESTY / "anatomia_wzorzec.png"), cv2.cvtColor(np.hstack(kadry), cv2.COLOR_RGB2BGR))

# 2) Przedramię
W, H = 600, 1300
fbo = ctx.simple_framebuffer((W, H), components=4)
kadry = []
for wnetrze, kat in ((True, None), (True, 0.0), (True, 35.0), (False, 0.0)):
    kadry.append(klatka(r, fbo, W, H, [dlon_wzorcowa(R, E, wnetrze, W, H, 2.4, kat, srodek_y=0.3)], WSZYSTKO))
cv2.imwrite(str(TESTY / "anatomia_przedramie.png"), cv2.cvtColor(np.hstack(kadry), cv2.COLOR_RGB2BGR))

# 3) Zdjęcie (łokieć dolnej ręki zaznaczony ręcznie: rękaw wychodzi za prawą krawędź zdjęcia)
rgb = cv2.cvtColor(cv2.imread(str(TESTY / "dlonie_test.jpg")), cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
H, W = rgb.shape[:2]
det, stab = dlonie.DetektorDloni(), stabilizacja.StabilizatorDloni("srednie")
for i in range(6):
    wynik = stab.aktualizuj(det.analizuj(rgb, lustro=False), i / 30, W, H)
det.zamknij()
print("dłonie:", [(d.strona, round(d.pewnosc, 2)) for d in wynik])
fbo = ctx.simple_framebuffer((W, H), components=4)
r.ustaw_obraz(np.ascontiguousarray(rgb).tobytes(), W, H)
dolna = max(wynik, key=lambda d: d.punkty[0][1])
x = np.asarray(dolna.punkty)[:, 0] * W
y = np.asarray(dolna.punkty)[:, 1] * H
x0, x1 = int(max(0, x.min() - 120)), int(min(W, x.max() + 700))
y0, y1 = int(max(0, y.min() - 120)), int(min(H, y.max() + 160))
kadry = [klatka(r, fbo, W, H, wynik, w)[y0:y1, x0:x1] for w in ({"kosci"}, WSZYSTKO)]
dolna.lokiec = (1.02, 0.735, 0.03)
kadry.append(klatka(r, fbo, W, H, wynik, WSZYSTKO)[y0:y1, x0:x1])
cv2.imwrite(str(TESTY / "anatomia_zdjecie.png"), cv2.cvtColor(np.vstack(kadry), cv2.COLOR_RGB2BGR))

t = time.perf_counter()
for _ in range(30):
    klatka(r, fbo, W, H, wynik, WSZYSTKO)
print(f"czas klatki {W}x{H} (z odczytem): {(time.perf_counter() - t) / 30 * 1000:.1f} ms")
