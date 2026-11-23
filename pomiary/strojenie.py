"""Szuka ustawień filtra One Euro: najmniejsze odstawanie w ruchu przy danym spokoju w bezruchu."""
import pickle
import sys

import numpy as np

sys.path.insert(0, r"A:\sledzenie-dloni")
import stabilizacja  # noqa: E402

FPS, A, OKRES, H, W = 30, 160, 1.5, 960, 640
SZER = W + 2 * A
dane = pickle.loads(open(sys.argv[1], "rb").read())


def po_pozycji(dl):
    out = {}
    for d in dl:
        p = np.asarray(d.punkty)[:, :2] * [SZER, H]
        out["gorna" if p[0, 1] < 0.6 * H else "dolna"] = p
    return out


sur = [po_pozycji(dl) for dl in dane["bezruch"]]
prawda = {k: np.mean([f[k] for f in sur[10:] if k in f], axis=0) for k in ("gorna", "dolna")}


def ocen(mc, beta):
    stabilizacja.POZIOMY["test"] = {"min_cutoff": mc, "beta": beta}
    w = {}
    for scen in ("bezruch", "ruch"):
        st = stabilizacja.StabilizatorDloni("test")
        w[scen] = [po_pozycji(st.aktualizuj(dl, i / FPS, SZER, H)) for i, dl in enumerate(dane[scen])]
    drg = np.mean([np.linalg.norm(a[k] - b[k], axis=1).mean()
                   for a, b in zip(w["bezruch"][10:], w["bezruch"][11:]) for k in prawda if k in a and k in b])
    odst = np.mean([np.linalg.norm(f[k] - (prawda[k] + [A * np.sin(2 * np.pi * (i / FPS) / OKRES), 0]), axis=1).mean()
                    for i, f in enumerate(w["ruch"]) if i >= 10 for k in prawda if k in f])
    return drg, odst


wyniki = []
for mc in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
    for beta in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3):
        wyniki.append((mc, beta, *ocen(mc, beta)))

# Dla kilku progów drgania: ustawienie z najmniejszym odstawaniem.
for prog in (0.15, 0.10, 0.07, 0.05):
    ok = [r for r in wyniki if r[2] <= prog]
    if ok:
        mc, beta, drg, odst = min(ok, key=lambda r: r[3])
        print(f"drganie <= {prog:.2f} px: min_cutoff={mc}, beta={beta} -> drganie {drg:.3f} px, odstawanie {odst:.2f} px")
