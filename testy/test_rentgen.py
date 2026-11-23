"""Test rentgena bez okna: zdjęcie -> MediaPipe -> stabilizacja -> rentgen (OpenGL) -> PNG w testy/."""
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
H, W = rgb.shape[:2]

det = dlonie.DetektorDloni()
stab = stabilizacja.StabilizatorDloni("srednie")
for i in range(6):
    wynik = stab.aktualizuj(det.analizuj(rgb, lustro=False), i / 30, W, H)
det.zamknij()
print("dłonie:", [(d.strona, round(d.pewnosc, 2)) for d in wynik])

# Czy osie pomiaru 3D zgadzają się z osiami obrazu?
for d in wynik:
    obraz = np.asarray(d.punkty)[:, :2] * [W, H]
    swiat = np.asarray(d.punkty_3d)
    kx = np.corrcoef(obraz[:, 0], swiat[:, 0])[0, 1]
    ky = np.corrcoef(obraz[:, 1], swiat[:, 1])[0, 1]
    print(f"{d.strona}: zgodność osi x={kx:+.2f}, y={ky:+.2f}  (+1 = te same kierunki)")
    P = rentgen.punkty_na_obrazie(d, W, H)
    print(f"   głębia punktów [px]: od {P[:, 2].min():.0f} do {P[:, 2].max():.0f}; "
          f"rozmiar dłoni {rentgen._rozmiar_dloni(P):.0f} px")

ctx = moderngl.create_standalone_context(require=330)
print("karta graficzna:", ctx.info["GL_RENDERER"])
fbo = ctx.simple_framebuffer((W, H), components=4)
fbo.use()
r = rentgen.RenderRentgen(ctx)
r.ustaw_obraz(np.ascontiguousarray(rgb).tobytes(), W, H)

for nazwa, warstwy in (("rentgen_kosci", {"kosci"}), ("rentgen_wszystko", {"kosci", "miesnie", "sciegna"})):
    fbo.clear(0, 0, 0, 1)
    r.rysuj(fbo, (0, 0, W, H), W, H, wynik, lustro=False, warstwy=warstwy, przyciemnienie=0.85)
    obraz = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)[::-1]
    cv2.imwrite(str(TESTY / f"{nazwa}.png"), cv2.cvtColor(obraz, cv2.COLOR_RGB2BGR))
    print("zapisano", nazwa)
