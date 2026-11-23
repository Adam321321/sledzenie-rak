"""
Przygotowanie modelu anatomicznego dłoni i przedramienia (uruchamiane raz, wynik: anatomia/dlon.npz).

Źródło: BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International.
Prawa ręka w pozycji anatomicznej, jednostki: milimetry.

Co powstaje:
- spoczynkowe położenie 21 punktów MediaPipe wyliczone z kształtu kości („wzorcowa dłoń”) i punkt łokcia,
- trzy warstwy siatek (kości, mięśnie, ścięgna) z wagami: każdy wierzchołek mówi, z którymi „kośćmi
  sterującymi” się porusza (19 kości palców i śródręcza, dłoń/nadgarstek jako całość, przedramię),
- dla każdego wierzchołka kierunek włókien (wzdłuż mięśnia, ścięgna albo kości) do rysowania tekstury
  i odległość od nadgarstka w stronę łokcia (do wygaszania przedramienia, gdy łokcia nie widać).
"""
from __future__ import annotations

import io
import logging
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

TU = Path(__file__).resolve().parent
sys.path.insert(0, str(TU))
import czesci  # noqa: E402

ZRODLO = TU / "zrodlo"
WYNIK = TU / "dlon.npz"

DLON = len(czesci.KOSCI_PALCOW)          # 19: dłoń/nadgarstek jako całość
PRZEDRAMIE = DLON + 1                    # 20: przedramię (od nadgarstka do łokcia)
LICZBA_STEROWANIA = PRZEDRAMIE + 1
ZA_LOKCIEM_MM = 45.0         # ile ponad staw łokciowy zostawić (przyczepy mięśni na kości ramiennej)
GRUBOSC_SCIEGNA_MM = 5.0     # cieńsze części mięśni przedramienia to ścięgna
PRZEJSCIE_NADGARSTKA = (-3.0, 15.0)   # mm od nadgarstka w stronę łokcia: tu tkanki przechodzą z dłoni na przedramię
OPUSZEK_MM = 3.0             # punkt MediaPipe na czubku palca leży kawałek za końcem kości


def gladka(m: trimesh.Trimesh) -> trimesh.Trimesh:
    v, f = trimesh.remesh.subdivide_loop(m.vertices, m.faces, iterations=1)
    return trimesh.Trimesh(v, f, process=True)


def najblizsze(cel: np.ndarray, zrodlo: np.ndarray) -> np.ndarray:
    """Dla każdego punktu z `cel` numer najbliższego punktu z `zrodlo`."""
    wynik = np.empty(len(cel), dtype=np.int64)
    for i in range(0, len(cel), 1024):
        d = ((cel[i:i + 1024, None, :] - zrodlo[None, :, :]) ** 2).sum(axis=2)
        wynik[i:i + 1024] = d.argmin(axis=1)
    return wynik


def grubosc(m: trimesh.Trimesh) -> np.ndarray:
    """
    Grubość bryły pod każdym wierzchołkiem: odległość do najbliższego wierzchołka po drugiej stronie
    (w stożku 35° wokół normalnej skierowanej do środka). Ścięgna mają kilka mm, brzuśce mięśni kilkanaście.
    (Szerszy stożek łapie sąsiednie wierzchołki tej samej powierzchni i wszystko wychodzi „cienkie”.)
    Gdzie w stożku nie ma żadnego wierzchołka (rzadka siatka na cienkich ścięgnach), wartość bierzemy
    od sąsiadów, zamiast uznać to miejsce za grube.
    """
    v, n = m.vertices, m.vertex_normals
    tri = v[m.faces]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:   # siatka „na lewą stronę”
        n = -n
    cos_stozka = np.cos(np.radians(35.0))
    wynik = np.full(len(v), np.nan)
    for i in range(0, len(v), 256):
        d = v[None, :, :] - v[i:i + 256, None, :]
        odl = np.linalg.norm(d, axis=2)
        wglab = -(d * n[i:i + 256, None, :]).sum(axis=2)
        odl = np.where((wglab > cos_stozka * odl) & (odl > 0.3), odl, np.inf).min(axis=1)
        wynik[i:i + 256] = np.where(np.isfinite(odl), np.minimum(odl, 60.0), np.nan)
    for _ in range(50):                    # dziury uzupełniane średnią znanych sąsiadów
        brak = np.isnan(wynik)
        if not brak.any():
            break
        suma, ile = np.zeros(len(v)), np.zeros(len(v))
        for a, b in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)):
            znane = ~np.isnan(wynik[m.faces[:, b]])
            np.add.at(suma, m.faces[znane, a], wynik[m.faces[znane, b]])
            np.add.at(ile, m.faces[znane, a], 1.0)
        wynik = np.where(brak & (ile > 0), suma / np.maximum(ile, 1.0), wynik)
    return np.nan_to_num(wynik, nan=60.0)


def wygladz_na_siatce(m: trimesh.Trimesh, wartosci: np.ndarray, razy: int = 2) -> np.ndarray:
    """Uśrednianie wartości z sąsiednimi wierzchołkami (przez wspólne trójkąty)."""
    w = wartosci.astype(float)
    for _ in range(razy):
        suma = np.zeros(len(w))
        ile = np.zeros(len(w))
        srednie = w[m.faces].mean(axis=1)
        for k in range(3):
            np.add.at(suma, m.faces[:, k], srednie)
            np.add.at(ile, m.faces[:, k], 1.0)
        w = np.where(ile > 0, suma / np.maximum(ile, 1.0), w)
    return w


def wczytaj_archiwum():
    pliki = defaultdict(list)
    for linia in (ZRODLO / "isa_element_parts.txt").read_text(encoding="utf-8").splitlines()[1:]:
        fma, _nazwa, fj = linia.split("\t")
        pliki[fma].append(fj)
    archiwum = zipfile.ZipFile(ZRODLO / "isa_BP3D_4.0_obj_99.zip")

    def siatka(fma: str, wygladzenie: int = 0, z_gruboscia: bool = False):
        """Siatka części (wszystkie jej pliki), opcjonalnie z grubością pod każdym wierzchołkiem."""
        kawalki, grubosci = [], []
        for fj in pliki[fma]:
            m = trimesh.load(io.BytesIO(archiwum.read(f"isa_BP3D_4.0_obj_99/{fj}.obj")), file_type="obj")
            m = trimesh.Trimesh(m.vertices, m.faces, process=True)   # sama geometria, wierzchołki scalone
            g = grubosc(m) if z_gruboscia else None                   # na surowej siatce (mniej punktów)
            surowe = m.vertices.copy()
            for _ in range(wygladzenie):   # każdy plik osobno: po sklejeniu dzielenie trójkątów się psuje
                m = gladka(m)
            kawalki.append(m)
            if z_gruboscia:
                grubosci.append(g[najblizsze(m.vertices, surowe)])
        m = trimesh.util.concatenate(kawalki)
        if not z_gruboscia:
            m.merge_vertices()
            return m
        return m, np.concatenate(grubosci)   # bez scalania: kolejność wierzchołków zgodna z grubością

    return siatka


def podsiatka(m: trimesh.Trimesh, maska_scian: np.ndarray, *atrybuty: np.ndarray):
    """Część siatki (wybrane trójkąty) razem z odpowiadającymi im wartościami na wierzchołkach."""
    f = m.faces[maska_scian]
    uzyte = np.unique(f)
    nowe = np.full(len(m.vertices), -1, dtype=np.int64)
    nowe[uzyte] = np.arange(len(uzyte))
    wynik = trimesh.Trimesh(m.vertices[uzyte], nowe[f], process=False)
    return (wynik, *[a[uzyte] for a in atrybuty]) if atrybuty else wynik


def os_glowna(v: np.ndarray) -> np.ndarray:
    """Najdłuższy kierunek bryły (PCA)."""
    return np.linalg.svd(v - v.mean(axis=0), full_matrices=False)[2][0]


def kawalki(m: trimesh.Trimesh) -> np.ndarray:
    """Numer spójnego kawałka dla każdego wierzchołka (np. 4 osobne mięśnie glistowate w jednym pliku)."""
    etykiety = np.arange(len(m.vertices))
    f = m.faces
    while True:
        nowe = etykiety.copy()
        np.minimum.at(nowe, f.ravel(), np.repeat(etykiety[f].min(axis=1), 3))
        nowe = nowe[nowe]                      # „skracanie ścieżek”: szybciej dochodzi do najmniejszej etykiety
        if np.array_equal(nowe, etykiety):
            return etykiety
        etykiety = nowe


def kierunek_kawalkami(m: trimesh.Trimesh) -> np.ndarray:
    """Kierunek włókien mięśnia: najdłuższa oś każdego spójnego kawałka osobno."""
    etykiety = kawalki(m)
    kier = np.zeros_like(m.vertices)
    for e in np.unique(etykiety):
        ktore = etykiety == e
        kier[ktore] = os_glowna(m.vertices[ktore]) if ktore.sum() >= 4 else (1.0, 0.0, 0.0)
    return kier.astype(np.float32)


def konce_kosci(v: np.ndarray, kierunek: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oś kości (PCA, zwrócona w stronę palców) oraz środki jej bliższego i dalszego końca."""
    srodek = v.mean(axis=0)
    os_ = os_glowna(v)
    if abs(np.dot(os_, kierunek)) < np.cos(np.radians(45)):  # krótkie, krępe kości: PCA bywa w poprzek
        os_ = kierunek
    if np.dot(os_, kierunek) < 0:
        os_ = -os_
    t = (v - srodek) @ os_
    blizszy = v[t <= np.percentile(t, 8)].mean(axis=0)
    dalszy = v[t >= np.percentile(t, 92)].mean(axis=0)
    return os_, blizszy, dalszy


def wzorcowe_punkty(kosci: list[trimesh.Trimesh], nadgarstek: dict[str, trimesh.Trimesh]) -> np.ndarray:
    """21 punktów MediaPipe na modelu anatomicznym."""
    R = np.zeros((21, 3))
    srodki = [k.vertices.mean(axis=0) for k in kosci]
    # Kości w KOSCI_PALCOW idą palcami: kciuk (3 kości), potem po 4 kości na palec.
    lancuchy = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]]
    for lancuch in lancuchy:
        konce = []
        for n, i in enumerate(lancuch):
            if n + 1 < len(lancuch):
                kierunek = srodki[lancuch[n + 1]] - srodki[i]
            else:
                kierunek = srodki[i] - srodki[lancuch[n - 1]]
            kierunek /= np.linalg.norm(kierunek)
            konce.append(konce_kosci(kosci[i].vertices, kierunek))
        for n, i in enumerate(lancuch):
            _fma, _opis, a, b = czesci.KOSCI_PALCOW[i]
            os_, blizszy, dalszy = konce[n]
            if n == 0 and a != 0:          # kciuk: staw u nasady I kości śródręcza
                R[a] = blizszy
            if n + 1 < len(lancuch):       # staw między tą a następną kością
                R[b] = (dalszy + konce[n + 1][1]) / 2
            else:                          # czubek palca
                R[b] = dalszy + os_ * OPUSZEK_MM
    # Nadgarstek: środek bliższego szeregu kości nadgarstka.
    R[0] = np.mean([nadgarstek[n].vertices.mean(axis=0) for n in ("FMA24435", "FMA24437", "FMA24439")], axis=0)
    return R


def os_dloni(R: np.ndarray) -> np.ndarray:
    return (R[9] - R[0]) / np.linalg.norm(R[9] - R[0])


def punkt_lokcia(promieniowa: trimesh.Trimesh, lokciowa: trimesh.Trimesh, R: np.ndarray) -> np.ndarray:
    """Środek stawu łokciowego: między szczytem głowy kości promieniowej a kością łokciową na tej wysokości."""
    w_gore = -os_dloni(R)                                 # od nadgarstka w stronę łokcia
    t = (promieniowa.vertices - R[0]) @ w_gore
    glowa = promieniowa.vertices[t >= np.percentile(t, 98)].mean(axis=0)
    poziom = float((glowa - R[0]) @ w_gore)
    tu = (lokciowa.vertices - R[0]) @ w_gore
    obok = lokciowa.vertices[np.abs(tu - poziom) < 4.0]
    return (glowa + (obok.mean(axis=0) if len(obok) else glowa)) / 2


def w_strone_lokcia(v: np.ndarray, R: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Ile mm od nadgarstka w stronę łokcia (ujemne = w dłoni)."""
    return (v - R[0]) @ ((E - R[0]) / np.linalg.norm(E - R[0]))


def odcinki_sterujace(R: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    odcinki = [(R[a], R[b]) for _f, _o, a, b in czesci.KOSCI_PALCOW]
    odcinki.append((R[0] - os_dloni(R) * 40.0, R[0]))  # dłoń/nadgarstek (i kawałek przedramienia)
    return odcinki


def odleglosc_do_odcinka(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _plynnie(x: np.ndarray, od: float, do: float) -> np.ndarray:
    x = np.clip((x - od) / (do - od), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def wagi(v: np.ndarray, R: np.ndarray, E: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dla mięśni i ścięgien: 4 najważniejsze kości sterujące i płynne wagi zależne od odległości.
    E (łokieć) podany = tkanka przedramienia: za nadgarstkiem przechodzi płynnie na kość „przedramię”.
    Zwraca też kierunek najbliższych kości (ścięgna biegną wzdłuż kości).
    """
    odcinki = odcinki_sterujace(R)
    d = np.stack([odleglosc_do_odcinka(v, a, b) for a, b in odcinki], axis=1)
    tau = 3.0  # mm: jak łagodnie przechodzi wpływ jednej kości w drugą
    w = np.exp(-(d - d.min(axis=1, keepdims=True)) / tau)
    w /= w.sum(axis=1, keepdims=True)
    kierunki = np.array([(b - a) / np.linalg.norm(b - a) for a, b in odcinki])
    kier = w @ kierunki
    W = np.zeros((len(v), LICZBA_STEROWANIA))
    W[:, :len(odcinki)] = w
    if E is not None:
        f = _plynnie(w_strone_lokcia(v, R, E), *PRZEJSCIE_NADGARSTKA)
        W *= (1.0 - f)[:, None]
        W[:, PRZEDRAMIE] += f
        kier = kier * (1.0 - f)[:, None] + np.outer(f, (E - R[0]) / np.linalg.norm(E - R[0]))
    kier /= np.linalg.norm(kier, axis=1, keepdims=True).clip(1e-9)
    idx = np.argsort(-W, axis=1)[:, :4]
    ww = np.take_along_axis(W, idx, axis=1)
    ww /= ww.sum(axis=1, keepdims=True)
    return idx.astype(np.float32), ww.astype(np.float32), kier.astype(np.float32)


def pakuj(siatki: list[tuple], R: np.ndarray, E: np.ndarray) -> dict[str, np.ndarray]:
    """Łączy siatki jednej warstwy w jedną tablicę wierzchołków i trójkątów."""
    pozycje, normalne, kosci_idx, wagi_, przedramie, kierunki, trojkaty = [], [], [], [], [], [], []
    przesuniecie = 0
    for m, idx, w, kier in siatki:
        pozycje.append(m.vertices.astype(np.float32))
        normalne.append(m.vertex_normals.astype(np.float32))
        kosci_idx.append(idx)
        wagi_.append(w)
        przedramie.append(w_strone_lokcia(m.vertices, R, E).astype(np.float32))
        kierunki.append(kier)
        trojkaty.append(m.faces.astype(np.uint32) + przesuniecie)
        przesuniecie += len(m.vertices)
    return {"pozycje": np.concatenate(pozycje), "normalne": np.concatenate(normalne),
            "kosci": np.concatenate(kosci_idx), "wagi": np.concatenate(wagi_),
            "przedramie": np.concatenate(przedramie)[:, None], "kierunek": np.concatenate(kierunki),
            "trojkaty": np.concatenate(trojkaty)}


def sztywne(m: trimesh.Trimesh, kosc: int, kierunek: np.ndarray) -> tuple:
    n = len(m.vertices)
    idx = np.zeros((n, 4), np.float32)
    idx[:, 0] = kosc
    w = np.zeros((n, 4), np.float32)
    w[:, 0] = 1.0
    kierunek = np.asarray(kierunek, np.float32)
    return m, idx, w, np.tile(kierunek / np.linalg.norm(kierunek), (n, 1))


def miekkie(m: trimesh.Trimesh, R: np.ndarray, E: np.ndarray | None, wlokna_wzdluz_kosci: bool) -> tuple:
    idx, w, kier_kosci = wagi(m.vertices, R, E)
    return m, idx, w, (kier_kosci if wlokna_wzdluz_kosci else kierunek_kawalkami(m))


def _wygladz(x: np.ndarray, okno: int = 5) -> np.ndarray:
    """Średnia ruchoma wzdłuż pierwszej osi (brzegi powielone, żeby nie ściągało końców)."""
    pol = okno // 2
    x_ = np.concatenate([np.repeat(x[:1], pol, axis=0), x, np.repeat(x[-1:], pol, axis=0)])
    return np.stack([x_[i:i + len(x)] for i in range(okno)]).mean(axis=0)


def rzut_na_lamana(v: np.ndarray, wezly: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dla punktów: położenie wzdłuż łamanej (mm od początku) i odległość od niej."""
    dl = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wezly, axis=0), axis=1))]
    odl = np.full(len(v), np.inf)
    s = np.zeros(len(v))
    for i in range(len(wezly) - 1):
        a, ab = wezly[i], wezly[i + 1] - wezly[i]
        t = np.clip(((v - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        d = np.linalg.norm(v - (a + t[:, None] * ab), axis=1)
        lepsze = d < odl
        odl[lepsze], s[lepsze] = d[lepsze], dl[i] + t[lepsze] * (dl[i + 1] - dl[i])
    return s, odl, dl


def pochewka(R: np.ndarray, lancuch: tuple[int, ...], v_sciegien: np.ndarray) -> tuple[trimesh.Trimesh, np.ndarray]:
    """
    Pochewka ścięgien zginaczy jednego palca: rurka owijająca ścięgna (przekrój mierzony co 1 mm),
    pogrubiona w miejscach więzadeł obrączkowatych (A1–A5), zwężona na końcach.
    Zwraca siatkę i kierunek „włókien” (w poprzek palca, więc tekstura układa się w obrączki).
    """
    kciuk = lancuch[0] != 0
    nasada, podstawa = R[lancuch[0]], R[lancuch[1]]
    ostatni_staw, czubek = R[lancuch[-2]], R[lancuch[-1]]
    poczatek = podstawa - (0.15 if kciuk else 0.18) * (podstawa - nasada)   # szyjka kości śródręcza
    koniec = ostatni_staw + 0.35 * (czubek - ostatni_staw)                   # nasada paliczka dalszego
    wezly = np.array([poczatek, *[R[i] for i in lancuch[1:-1]], koniec])
    s_v, odl, dl = rzut_na_lamana(v_sciegien, wezly)
    s = np.arange(0.0, dl[-1] + 1e-6, 1.0)

    # Oś i układ odniesienia w każdym przekroju.
    seg = np.clip(np.searchsorted(dl, s, side="right") - 1, 0, len(wezly) - 2)
    t = (s - dl[seg]) / (dl[seg + 1] - dl[seg])
    os_pkt = wezly[seg] + t[:, None] * (wezly[seg + 1] - wezly[seg])
    T = _wygladz((wezly[seg + 1] - wezly[seg]) / (dl[seg + 1] - dl[seg])[:, None], 7)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    gora = np.cross(R[2] - R[1], R[5] - R[1]) if kciuk else np.cross(R[5] - R[0], R[17] - R[0])
    N = gora - (T @ gora)[:, None] * T
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    B = np.cross(T, N)

    # Przekrój ścięgien: środek i półosie elipsy w każdym przekroju (brakujące dopełniane z sąsiednich).
    srodki = np.full((len(s), 3), np.nan)
    polosie = np.full((len(s), 2), np.nan)
    for j in range(len(s)):
        ktore = (np.abs(s_v - s[j]) <= 1.0) & (odl < 8.0)
        if ktore.sum() >= 6:
            q = v_sciegien[ktore] - os_pkt[j]
            b, n = q @ B[j], q @ N[j]
            srodki[j] = os_pkt[j] + B[j] * (b.max() + b.min()) / 2 + N[j] * (n.max() + n.min()) / 2
            polosie[j] = ((b.max() - b.min()) / 2, (n.max() - n.min()) / 2)
    znane = ~np.isnan(polosie[:, 0])
    for tab in (srodki, polosie):
        for k in range(tab.shape[1]):
            tab[:, k] = np.interp(s, s[znane], tab[znane, k])
    # Przy przyczepie zginacza powierzchownego jego pasma rozchodzą się na boki; pochewka nie rośnie razem z nimi.
    polosie = np.minimum(polosie, 1.15 * np.median(polosie, axis=0))
    srodki, polosie = _wygladz(srodki, 5), _wygladz(polosie, 7)
    polosie = np.maximum(polosie + 0.5, 1.2)           # trochę luzu wokół ścięgien

    # Więzadła obrączkowate: (środek, połowa szerokości) w mm wzdłuż pochewki.
    st = dl[1:-1]
    if kciuk:
        obraczki = [(st[0], 3.5), ((st[0] + st[1]) / 2, 0.18 * (st[1] - st[0])), (st[1], 2.5)]
    else:
        obraczki = [(st[0], 4.0), (st[0] + 0.35 * (st[1] - st[0]), 0.2 * (st[1] - st[0])), (st[1], 2.5),
                    ((st[1] + st[2]) / 2, 0.2 * (st[2] - st[1])), (st[2], 2.0)]
    garb = np.max([np.exp(-((s - c) / hw) ** 6) for c, hw in obraczki], axis=0)
    mnoznik = (1.0 + 0.14 * garb) * (0.8 + 0.2 * _plynnie(s, 0.0, 3.0) * _plynnie(dl[-1] - s, 0.0, 3.0))
    polosie *= mnoznik[:, None]

    K = 18
    kat = np.linspace(0.0, 2 * np.pi, K, endpoint=False)
    v = (srodki[:, None, :] + polosie[:, None, 0:1] * np.cos(kat)[None, :, None] * B[:, None, :]
         + polosie[:, None, 1:2] * np.sin(kat)[None, :, None] * N[:, None, :]).reshape(-1, 3)
    j, k = np.meshgrid(np.arange(len(s) - 1), np.arange(K), indexing="ij")
    i0, i1 = j * K + k, j * K + (k + 1) % K
    i2, i3 = i0 + K, i1 + K
    f = np.concatenate([np.stack([i0, i2, i1], -1).reshape(-1, 3), np.stack([i1, i2, i3], -1).reshape(-1, 3)])
    kier = np.repeat(B, K, axis=0).astype(np.float32)
    return trimesh.Trimesh(v, f, process=False), kier


def main() -> None:
    logging.getLogger("trimesh").setLevel(logging.ERROR)  # bez scipy liczy normalne wolniejszą drogą; to nie błąd
    siatka = wczytaj_archiwum()
    print("Wczytuję kości…")
    kosci = [siatka(fma) for fma, *_ in czesci.KOSCI_PALCOW]
    nadgarstek = {fma: siatka(fma) for fma, _ in czesci.KOSCI_NADGARSTKA}
    R = wzorcowe_punkty(kosci, nadgarstek)
    przedramie = {fma: siatka(fma, 2) for fma, _ in czesci.KOSCI_PRZEDRAMIENIA}   # w źródle dość kanciaste
    E = punkt_lokcia(*przedramie.values(), R)
    dlugosc = float(np.linalg.norm(E - R[0]))
    print(f"  przedramię (nadgarstek–łokieć): {dlugosc:.0f} mm, dłoń (nadgarstek–nasada środkowego): "
          f"{np.linalg.norm(R[9] - R[0]):.0f} mm")

    print("Warstwa: kości")
    warstwa_kosci = [sztywne(gladka(k), i, R[b] - R[a])
                     for i, (k, (_f, _o, a, b)) in enumerate(zip(kosci, czesci.KOSCI_PALCOW))]
    warstwa_kosci += [sztywne(gladka(k), DLON, os_glowna(k.vertices)) for k in nadgarstek.values()]
    warstwa_kosci += [sztywne(k, PRZEDRAMIE, E - R[0]) for k in przedramie.values()]

    print("Warstwa: mięśnie dłoni")
    warstwa_miesni = [miekkie(siatka(fma, 1), R, None, False) for fma, _opis in czesci.MIESNIE_DLONI]

    print("Mięśnie przedramienia (brzusiec / ścięgno):")
    warstwa_sciegien = []
    calosci = {}
    for fma, opis, opcje in czesci.MIESNIE_PRZEDRAMIENIA:
        m, g = siatka(fma, 1, z_gruboscia=True)
        x = w_strone_lokcia(m.vertices, R, E)[m.faces]
        maska = (x <= dlugosc + ZA_LOKCIEM_MM).all(axis=1)
        if "do_nadgarstka" in opcje:
            maska &= (x >= 0.0).all(axis=1)
        m, g = podsiatka(m, maska, g)
        calosci[fma] = m
        g = wygladz_na_siatce(m, np.minimum(g, 20.0), razy=4)   # bez pojedynczych plamek
        # W dłoni (dalej niż 10 mm za nadgarstkiem) mięśnie przedramienia mają już tylko ścięgna.
        w_dloni = w_strone_lokcia(m.vertices, R, E)[m.faces].mean(axis=1) < -10.0
        sciegno = (np.zeros(len(m.faces), bool) if "brzusiec" in opcje
                   else (g[m.faces].mean(axis=1) < GRUBOSC_SCIEGNA_MM) | w_dloni)
        if (~sciegno).any():
            warstwa_miesni.append(miekkie(podsiatka(m, ~sciegno), R, E, False))
        if sciegno.any():
            warstwa_sciegien.append(miekkie(podsiatka(m, sciegno), R, E, True))
        print(f"  {opis:45s} ścięgno {sciegno.mean():4.0%}   grubość: mediana {np.median(g):4.1f} mm")

    for fma, _opis in czesci.TROCZKI:          # troczek: włókna w poprzek nadgarstka
        warstwa_sciegien.append(miekkie(siatka(fma, 1), R, None, False))
    print("Pochewki ścięgien na palcach")
    for lancuch, fma_sciegien in czesci.POCHEWKI:
        m, kier = pochewka(R, lancuch, np.concatenate([calosci[f].vertices for f in fma_sciegien]))
        idx, w, _ = wagi(m.vertices, R, None)
        warstwa_sciegien.append((m, idx, w, kier))

    dane = {"punkty": R.astype(np.float32), "lokiec": E.astype(np.float32)}
    for nazwa, warstwa in (("kosci", warstwa_kosci), ("miesnie", warstwa_miesni), ("sciegna", warstwa_sciegien)):
        for klucz, tablica in pakuj(warstwa, R, E).items():
            dane[f"{nazwa}_{klucz}"] = tablica
        p, t = dane[f"{nazwa}_pozycje"], dane[f"{nazwa}_trojkaty"]
        krawedz = float(np.linalg.norm(p[t] - p[np.roll(t, 1, axis=1)], axis=2).max())
        print(f"  {nazwa}: {len(p):,} wierzchołków, {len(t):,} trójkątów, najdłuższa krawędź {krawedz:.1f} mm")
        if krawedz > 60.0:   # trzon kości promieniowej ma ok. 40 mm; zepsute siatki miały ponad 1000 mm
            raise SystemExit(f"Podejrzanie długie trójkąty w warstwie {nazwa} — siatka się popsuła.")
    np.savez_compressed(WYNIK, **dane)
    print(f"Zapisano {WYNIK} ({WYNIK.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
