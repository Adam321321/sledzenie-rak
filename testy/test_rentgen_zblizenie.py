"""Rentgen w powiększeniu (zdjęcie ×3), wycinek dolnej dłoni: kości osobno i wszystkie warstwy obok siebie."""
import sys
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
rgb = cv2.cvtColor(cv2.imread(str(TESTY / "dlonie_test.jpg")), cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
H, W = rgb.shape[:2]
det, stab = dlonie.DetektorDloni(), stabilizacja.StabilizatorDloni("srednie")
for i in range(6):
    wynik = stab.aktualizuj(det.analizuj(rgb, lustro=False), i / 30, W, H)
det.zamknij()

ctx = moderngl.create_standalone_context(require=330)
fbo = ctx.simple_framebuffer((W, H), components=4)
fbo.use()
r = rentgen.RenderRentgen(ctx)
r.ustaw_obraz(np.ascontiguousarray(rgb).tobytes(), W, H)
dolna = max(wynik, key=lambda d: d.punkty[0][1])
x = np.asarray(dolna.punkty)[:, 0] * W
y = np.asarray(dolna.punkty)[:, 1] * H
x0, x1 = int(max(0, x.min() - 120)), int(min(W, x.max() + 220))
y0, y1 = int(max(0, y.min() - 120)), int(min(H, y.max() + 120))
kadry = []
for warstwy in ({"kosci"}, {"kosci", "miesnie", "sciegna"}):
    fbo.clear(0, 0, 0, 1)
    r.rysuj(fbo, (0, 0, W, H), W, H, wynik, lustro=False, warstwy=warstwy, przyciemnienie=0.85)
    obraz = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)[::-1]
    kadry.append(obraz[y0:y1, x0:x1])
cv2.imwrite(str(TESTY / "rentgen_zblizenie.png"), cv2.cvtColor(np.hstack(kadry), cv2.COLOR_RGB2BGR))
print("zapisano", np.hstack(kadry).shape)
