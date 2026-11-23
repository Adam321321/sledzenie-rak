"""Rzut oka na modele z BodyParts3D: rozmiary, jednostki, osie."""
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "anatomia"))
import czesci  # noqa: E402

ZRODLO = BASE / "anatomia" / "zrodlo"
pliki = defaultdict(list)
for linia in (ZRODLO / "isa_element_parts.txt").read_text(encoding="utf-8").splitlines()[1:]:
    fma, _nazwa, fj = linia.split("\t")
    pliki[fma].append(fj)

archiwum = zipfile.ZipFile(ZRODLO / "isa_BP3D_4.0_obj_99.zip")


def wczytaj(fma):
    czesci_ = [trimesh.load(io.BytesIO(archiwum.read(f"isa_BP3D_4.0_obj_99/{fj}.obj")), file_type="obj", process=False)
               for fj in pliki[fma]]
    return trimesh.util.concatenate(czesci_)


for fma, opis, a, b in czesci.KOSCI_PALCOW:
    m = wczytaj(fma)
    srodek = m.bounds.mean(axis=0)
    print(f"{opis:<32} plików={len(pliki[fma])} wierzch.={len(m.vertices):>5} trójk.={len(m.faces):>5} "
          f"środek={np.round(srodek, 1)} wymiary={np.round(m.extents, 1)}")
for fma, opis, *_ in czesci.MIESNIE_DLONI[:2] + czesci.SCIEGNA[:2]:
    m = wczytaj(fma)
    print(f"{opis:<32} plików={len(pliki[fma])} wierzch.={len(m.vertices):>5} z od {m.bounds[0][2]:.0f} do {m.bounds[1][2]:.0f}")
