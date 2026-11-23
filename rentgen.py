"""
Rentgen 3D dłoni: anatomiczny model (kości, mięśnie, ścięgna) dopasowany do śledzonych punktów
i narysowany przez kartę graficzną (OpenGL, biblioteka moderngl).

Model: BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International
(przygotowany skryptem anatomia/przygotuj.py).

Jak to działa:
- Ze wzorcowej dłoni (anatomia/dlon.npz) znamy spoczynkowe położenie 21 punktów MediaPipe.
- Dla każdej kości liczymy przekształcenie „wzorcowa → twoja”: obrót + rozciągnięcie wzdłuż kości
  + skala grubości. Kości są sztywne; mięśnie i ścięgna mieszają przekształcenia kilku kości (wagi).
- x, y punktów bierzemy prosto z obrazu (model leży dokładnie na palcach), a głębię z pomiaru 3D
  MediaPipe (w metrach), przeskalowanego do pikseli.
- Przedramię ma własną kość sterującą: od nadgarstka do łokcia (z modelu ciała, lokcie.py), obróconą
  razem z dłonią. Gdy łokcia nie widać, przedramię trzyma się dłoni i łagodnie zanika za nadgarstkiem.
"""
from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np

from dlonie import Dlon

MODEL = Path(__file__).resolve().parent / "anatomia" / "dlon.npz"
LICZBA_STEROWANIA = 21      # 19 kości palców i śródręcza + dłoń/nadgarstek + przedramię
DLON = 19
PRZEDRAMIE = 20
ZANIK_BEZ_LOKCIA = (8.0, 45.0)    # mm od nadgarstka w stronę łokcia: od-do znikania, gdy łokcia nie widać
ZANIK_ZA_LOKCIEM = (20.0, 45.0)   # mm za stawem łokciowym (przyczepy mięśni na kości ramiennej)
KOSCI_PALCOW = [  # (punkt bliższy, punkt dalszy) dla kości sterujących 0–18, jak w anatomia/czesci.py
    (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
PUNKTY_DLONI = [0, 1, 5, 9, 13, 17]
WARSTWY = ("miesnie", "sciegna", "kosci")  # kolejność rysowania
KOLORY = {  # rentgen: (kolor, jasność, ile świeci środek względem krawędzi)
    "kosci": ((0.80, 0.92, 1.00), 1.00, 0.16),
    "miesnie": ((1.00, 0.28, 0.22), 0.45, 0.06),
    "sciegna": ((1.00, 0.88, 0.62), 0.55, 0.08),
}
MATERIALY = {"kosci": 0, "miesnie": 1, "sciegna": 2}   # anatomia: numer materiału w shaderze
STYLE = ("rentgen", "anatomia")
PROBKI_MSAA = 4                                       # wygładzanie krawędzi w widoku Anatomia


# --------------------------------------------------------------------------- geometria (czyste numpy)

def _norm(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _rama(P: np.ndarray, a: int, b: int, kciuk: bool) -> tuple[np.ndarray, float]:
    """Układ osi kości: x wzdłuż kości, y w stronę „góry” dłoni (normalna dłoni albo płaszczyzny kciuka)."""
    u = P[b] - P[a]
    dlugosc = float(np.linalg.norm(u))
    u = u / max(dlugosc, 1e-9)
    gora = np.cross(P[5] - P[0], P[17] - P[0])
    if kciuk:
        gora_kciuka = np.cross(P[2] - P[1], P[5] - P[1])
        if np.linalg.norm(gora_kciuka) > 1e-6 * max(1.0, np.linalg.norm(gora)):
            gora = gora_kciuka
    v = gora - np.dot(gora, u) * u
    if np.linalg.norm(v) < 1e-9:  # kość prostopadła do „góry”: dowolny stabilny kierunek
        v = np.cross(u, [0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0.0, 0.0])
    v = _norm(v)
    return np.stack([u, v, np.cross(u, v)], axis=1), dlugosc


def _rozmiar_dloni(P: np.ndarray) -> float:
    return (np.linalg.norm(P[9] - P[0]) + np.linalg.norm(P[17] - P[5])) / 2.0


def _podobienstwo(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Umeyama: obrót + jednakowa skala + przesunięcie najlepiej przenoszące punkty X na Y (bez odbić)."""
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    Rm = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / max(float((Xc ** 2).sum() / len(X)), 1e-12)
    M = np.eye(4)
    M[:3, :3] = s * Rm
    M[:3, 3] = my - s * Rm @ mx
    return M


def _rama_przedramienia(P: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Układ osi przedramienia: x od nadgarstka do łokcia, y w poprzek dłoni (od małego palca do wskazującego).
    Oś w poprzek dłoni nie zmienia się przy zginaniu nadgarstka, a obraca się z dłonią przy nawracaniu
    i odwracaniu przedramienia, więc przedramię „skręca się” razem z dłonią.
    """
    u = E - P[0]
    dlugosc = float(np.linalg.norm(u))
    u = u / max(dlugosc, 1e-9)
    bok = P[5] - P[17]
    v = bok - np.dot(bok, u) * u
    if np.linalg.norm(v) < 1e-6 * max(1.0, float(np.linalg.norm(bok))):
        gora = np.cross(P[5] - P[0], P[17] - P[0])
        v = gora - np.dot(gora, u) * u
    v = _norm(v)
    return np.stack([u, v, np.cross(u, v)], axis=1), dlugosc


def poza(P: np.ndarray, R: np.ndarray, Pe: np.ndarray | None = None, Re: np.ndarray | None = None) -> np.ndarray:
    """
    P: punkty twojej dłoni (21×3, piksele), R: punkty wzorcowej dłoni (21×3, mm).
    Pe, Re: twój łokieć (piksele; None = nie widać) i łokieć wzorcowy (mm).
    Zwraca 21 macierzy 4×4 przenoszących model z układu wzorcowego na obraz.
    """
    M = np.tile(np.eye(4), (LICZBA_STEROWANIA, 1, 1))
    skala = _rozmiar_dloni(P) / max(_rozmiar_dloni(R), 1e-9)
    for k, (a, b) in enumerate(KOSCI_PALCOW):
        kciuk = k < 3
        F_r, d_r = _rama(R, a, b, kciuk)
        F_p, d_p = _rama(P, a, b, kciuk)
        A = F_p @ np.diag([d_p / max(d_r, 1e-9), skala, skala]) @ F_r.T
        M[k, :3, :3] = A
        M[k, :3, 3] = P[a] - A @ R[a]
    M[DLON] = _podobienstwo(R[PUNKTY_DLONI], P[PUNKTY_DLONI])
    if Pe is None or Re is None:
        M[PRZEDRAMIE] = M[DLON]          # bez łokcia: przedramię przyklejone do dłoni (i zanika)
        return M
    # Głębia łokcia z modelu ciała bywa przesadzona: przedramię nie może być dłuższe niż ~1,25 wzorcowego.
    Pe = np.array(Pe, dtype=float)
    d_xy = float(np.linalg.norm(Pe[:2] - P[0, :2]))
    d_max = 1.25 * float(np.linalg.norm(Re - R[0])) * skala
    dz_max = float(np.sqrt(max(d_max ** 2 - d_xy ** 2, 0.0)))
    Pe[2] = P[0, 2] + float(np.clip(Pe[2] - P[0, 2], -dz_max, dz_max))
    F_r, d_r = _rama_przedramienia(R, Re)
    F_p, d_p = _rama_przedramienia(P, Pe)
    A = F_p @ np.diag([d_p / max(d_r, 1e-9), skala, skala]) @ F_r.T
    M[PRZEDRAMIE, :3, :3] = A
    M[PRZEDRAMIE, :3, 3] = P[0] - A @ R[0]
    return M


def punkty_na_obrazie(d: Dlon, szer: int, wys: int) -> tuple[np.ndarray, np.ndarray | None]:
    """
    21 punktów w pikselach: x, y z obrazu, głębia z pomiaru 3D MediaPipe przeskalowana do pikseli.
    Do tego łokieć w tym samym układzie (albo None, gdy go nie widać).
    """
    obraz = np.asarray(d.punkty, dtype=float)[:, :2] * [szer, wys]
    swiat = np.asarray(d.punkty_3d, dtype=float)
    o_c = obraz - obraz.mean(axis=0)
    s_c = swiat[:, :2] - swiat[:, :2].mean(axis=0)
    skala = float((o_c * s_c).sum() / max((s_c ** 2).sum(), 1e-12))  # piksele na metr
    glebia = (swiat[:, 2] - swiat[0, 2]) * abs(skala)
    lokiec = None
    if d.lokiec is not None:
        x, y, dz = d.lokiec
        lokiec = np.array([x * szer, y * wys, dz * abs(skala)])
    return np.column_stack([obraz, glebia]), lokiec


# --------------------------------------------------------------------------- rysowanie (OpenGL)

_TLO_VS = """
#version 330
in vec2 in_xy;
out vec2 uv;
void main() { uv = vec2(in_xy.x * 0.5 + 0.5, 0.5 - in_xy.y * 0.5); gl_Position = vec4(in_xy, 0.0, 1.0); }
"""
_TLO_FS = """
#version 330
uniform sampler2D obraz;
uniform sampler2D swiatlo;     // zsumowane światło anatomii (bufor zmiennoprzecinkowy)
uniform float przyciemnienie;
uniform float ekspozycja;
uniform int jest_obraz;
in vec2 uv;
out vec4 kolor;
void main() {
    vec3 c = jest_obraz == 1 ? texture(obraz, uv).rgb : vec3(0.0);
    float szary = dot(c, vec3(0.299, 0.587, 0.114));
    vec3 rtg = mix(c, vec3(0.55, 0.75, 1.0) * szary, przyciemnienie);   // niebieskawy, jak klisza
    rtg *= 1.0 - 0.75 * przyciemnienie;
    // „Tone mapping”: łagodne ściskanie jasności, więc nakładające się warstwy nie przepalają się na biało.
    vec3 s = 1.0 - exp(-texture(swiatlo, vec2(uv.x, 1.0 - uv.y)).rgb * ekspozycja);
    kolor = vec4(rtg + s * (1.0 - rtg), 1.0);
}
"""
_ANAT_VS = """
#version 330
uniform mat4 kosci[21];
uniform mat4 rzut;
uniform vec2 zanik;            // od-do (mm od nadgarstka w stronę łokcia), gdzie przedramię znika
in vec3 in_poz;
in vec3 in_norm;
in vec4 in_kosci;
in vec4 in_wagi;
in float in_przedramie;
out vec3 v_norm;
out float v_zanik;
void main() {
    v_zanik = 1.0 - smoothstep(zanik.x, zanik.y, in_przedramie);
    mat4 M = kosci[int(in_kosci.x)] * in_wagi.x + kosci[int(in_kosci.y)] * in_wagi.y
           + kosci[int(in_kosci.z)] * in_wagi.z + kosci[int(in_kosci.w)] * in_wagi.w;
    v_norm = mat3(M) * in_norm;
    gl_Position = rzut * (M * vec4(in_poz, 1.0));
}
"""
_ANAT_FS = """
#version 330
uniform vec3 barwa;
uniform float jasnosc;
uniform float srodek;
in vec3 v_norm;
in float v_zanik;
out vec4 kolor;
void main() {
    float zwrot = abs(normalize(v_norm).z);          // 1 = powierzchnia zwrócona do kamery
    float krawedz = pow(1.0 - zwrot, 2.0);           // na krawędziach świeci mocniej, jak na kliszy RTG
    kolor = vec4(barwa * (srodek + (1.0 - srodek) * krawedz) * jasnosc * v_zanik, 1.0);
}
"""

# ---- styl „Anatomia”: pełne, oświetlone bryły jak na rycinie z atlasu anatomii
_ATLAS_VS = """
#version 330
uniform mat4 kosci[21];
uniform mat4 rzut;
uniform vec2 zanik;
in vec3 in_poz;
in vec3 in_norm;
in vec4 in_kosci;
in vec4 in_wagi;
in float in_przedramie;
in vec3 in_kier;
out vec3 v_norm;
out vec3 v_mpoz;
out vec3 v_mkier;
out float v_zanik;
void main() {
    mat4 M = kosci[int(in_kosci.x)] * in_wagi.x + kosci[int(in_kosci.y)] * in_wagi.y
           + kosci[int(in_kosci.z)] * in_wagi.z + kosci[int(in_kosci.w)] * in_wagi.w;
    mat3 A = mat3(M);
    // Normalne przy nierównym rozciąganiu: macierz odwrotna transponowana (gdy się da).
    v_norm = abs(determinant(A)) > 1e-9 ? transpose(inverse(A)) * in_norm : A * in_norm;
    v_mpoz = in_poz;      // tekstura „przyklejona” do modelu (w mm), więc nie pływa przy ruchu
    v_mkier = in_kier;
    v_zanik = 1.0 - smoothstep(zanik.x, zanik.y, in_przedramie);
    gl_Position = rzut * (M * vec4(in_poz, 1.0));
}
"""
_ATLAS_FS = """
#version 330
uniform int material;          // 0 kości, 1 mięśnie, 2 ścięgna
uniform float widocznosc;
in vec3 v_norm;
in vec3 v_mpoz;
in vec3 v_mkier;
in float v_zanik;
out vec4 kolor;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.11, 0.17, 0.13));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float szum(vec3 x) {           // płynny szum wartości 0..1
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x),
                   mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
                   mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
// Włókna: szum mocno rozciągnięty wzdłuż kierunku włókien (prążki jak w mięśniu albo ścięgnie).
float wlokna(float wzdluz, float wpoprzek) {
    vec3 f = normalize(v_mkier);
    vec3 a = normalize(cross(f, abs(f.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
    vec3 b = cross(f, a);
    vec3 q = vec3(dot(v_mpoz, f) * wzdluz, dot(v_mpoz, a) * wpoprzek, dot(v_mpoz, b) * wpoprzek);
    float w = szum(q) * 0.55 + szum(q * 2.07 + 7.3) * 0.30 + szum(q * 4.3 + 3.1) * 0.15;
    // Gdy prążki robią się drobniejsze niż piksel, wygaszamy je (bez migotania i „mory”).
    float ostrosc = clamp(1.6 - 1.4 * length(fwidth(q.yz)), 0.0, 1.0);
    return mix(0.5, w, ostrosc);
}
void main() {
    vec3 n = normalize(v_norm);
    if (n.z > 0.0) n = -n;                         // kamera patrzy wzdłuż +z: normalna do kamery ma z < 0
    vec3 V = vec3(0.0, 0.0, -1.0);
    vec3 L1 = normalize(vec3(-0.35, -0.55, -0.76)); // główne światło: z góry, z lewej, od przodu
    vec3 L2 = normalize(vec3(0.60, 0.30, -0.74));   // słabsze, wypełniające
    float d1 = max(dot(n, L1), 0.0);
    float d2 = max(dot(n, L2), 0.0);
    float nh = max(dot(n, normalize(L1 + V)), 0.0);
    float zwrot = -n.z;

    vec3 baza, cien;
    float polysk, twardosc;
    if (material == 1) {               // mięsień: czerwony, wyraźne włókna, wilgotny połysk
        float w = wlokna(0.12, 1.5);
        baza = mix(vec3(0.62, 0.19, 0.17), vec3(0.96, 0.54, 0.48), w);
        cien = vec3(0.46, 0.20, 0.24);
        polysk = 0.20; twardosc = 24.0;
    } else if (material == 2) {        // ścięgno: perłowobiałe, drobne podłużne włókna, satynowy połysk
        float w = wlokna(0.07, 2.4);
        baza = mix(vec3(0.80, 0.82, 0.87), vec3(0.97, 0.97, 0.98), w);
        cien = vec3(0.52, 0.56, 0.70);
        polysk = 0.42; twardosc = 40.0;
    } else {                           // kość: kremowa, lekko nierówna
        float w = szum(v_mpoz * 0.5) * 0.6 + szum(v_mpoz * 1.7 + 4.0) * 0.4;
        baza = mix(vec3(0.83, 0.72, 0.52), vec3(0.96, 0.90, 0.75), w);
        cien = vec3(0.60, 0.50, 0.42);
        polysk = 0.10; twardosc = 14.0;
    }
    vec3 c = baza * (cien * 0.55 + vec3(0.62) * d1 + vec3(0.22) * d2) + vec3(polysk * pow(nh, twardosc));
    c *= mix(0.45, 1.0, smoothstep(0.0, 0.6, zwrot));   // ciemniejsze brzegi: bryły odcinają się od siebie
    float a = clamp(v_zanik * widocznosc, 0.0, 1.0);
    kolor = vec4(c * a, a);                               // kolor przemnożony przez krycie
}
"""
_ATLAS_TLO_FS = """
#version 330
uniform sampler2D obraz;
uniform sampler2D anatomia;    // RGBA, kolor przemnożony przez krycie
uniform float przyciemnienie;
uniform int jest_obraz;
uniform vec2 piksel;           // rozmiar piksela anatomii w jednostkach tekstury
in vec2 uv;
out vec4 kolor;
void main() {
    vec3 c = jest_obraz == 1 ? texture(obraz, uv).rgb : vec3(0.0);
    c *= 1.0 - przyciemnienie;
    vec2 ua = vec2(uv.x, 1.0 - uv.y);
    vec4 a = texture(anatomia, ua);
    // Miękki cień pod modelem (lekko w prawo i w dół): model „leży” na dłoni, a nie jest naklejką.
    float s = 0.0;
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++)
            s += texture(anatomia, ua + piksel * (vec2(-4.0, 5.0) + 3.0 * vec2(i, j))).a;
    c *= 1.0 - 0.5 * s / 25.0;
    kolor = vec4(c * (1.0 - a.a) + a.rgb, 1.0);
}
"""


class RentgenPozaEkranem:
    """
    Rentgen we własnym kontekście OpenGL (niezależnym od okna): rysuje obraz z kamery + anatomię
    i oddaje gotową klatkę jako tablicę RGB. Używać z jednego wątku (tego, który go utworzył).
    """

    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330)
        self.render = RenderRentgen(self.ctx)
        self._fbo: moderngl.Framebuffer | None = None
        self.karta = self.ctx.info.get("GL_RENDERER", "?")

    def klatka(self, obraz: bytes | memoryview, szer: int, wys: int, wyrownanie: int, nowy_obraz: bool,
               dlonie: list[Dlon], lustro: bool, warstwy: set[str], cel_szer: int, cel_wys: int,
               styl: str = "rentgen") -> np.ndarray:
        """Zwraca obraz cel_wys×cel_szer×3 (uint8, RGB, wiersze od góry)."""
        if self._fbo is None or self._fbo.size != (cel_szer, cel_wys):
            if self._fbo is not None:
                self._fbo.release()
            self._fbo = self.ctx.simple_framebuffer((cel_szer, cel_wys), components=3)
        if nowy_obraz:
            self.render.ustaw_obraz(obraz, szer, wys, wyrownanie)
        przyciemnienie = 0.85 if styl == "rentgen" else 0.25
        self.render.rysuj(self._fbo, (0, 0, cel_szer, cel_wys), szer, wys, dlonie, lustro, warstwy,
                          przyciemnienie, styl=styl)
        dane = self._fbo.read(components=3, alignment=1)
        return np.frombuffer(dane, dtype=np.uint8).reshape(cel_wys, cel_szer, 3)[::-1]


class ModelAnatomiczny:
    def __init__(self, sciezka: Path = MODEL):
        dane = np.load(sciezka)
        self.punkty = dane["punkty"].astype(float)
        self.lokiec = dane["lokiec"].astype(float)
        self.dlugosc_przedramienia = float(np.linalg.norm(self.lokiec - self.punkty[0]))
        self.warstwy = {w: {k: dane[f"{w}_{k}"]
                            for k in ("pozycje", "normalne", "kosci", "wagi", "przedramie", "kierunek", "trojkaty")}
                        for w in WARSTWY}


class RenderRentgen:
    """Rysuje tło (obraz z kamery) i rentgen dłoni. Działa w dowolnym kontekście moderngl."""

    def __init__(self, ctx: moderngl.Context, model: ModelAnatomiczny | None = None):
        self.ctx = ctx
        self.model = model or ModelAnatomiczny()
        self._tlo = ctx.program(vertex_shader=_TLO_VS, fragment_shader=_TLO_FS)
        prostokat = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._tlo_vao = ctx.vertex_array(self._tlo, [(prostokat, "2f", "in_xy")])
        self._tekstura: moderngl.Texture | None = None
        self._anat = ctx.program(vertex_shader=_ANAT_VS, fragment_shader=_ANAT_FS)
        self._atlas = ctx.program(vertex_shader=_ATLAS_VS, fragment_shader=_ATLAS_FS)
        self._atlas_tlo = ctx.program(vertex_shader=_TLO_VS, fragment_shader=_ATLAS_TLO_FS)
        self._atlas_tlo_vao = ctx.vertex_array(self._atlas_tlo, [(prostokat, "2f", "in_xy")])
        # Dwie wersje modelu: prawa dłoń i jej lustrzane odbicie (lewa); każda dla obu stylów.
        self._vao: dict[tuple[str, bool], moderngl.VertexArray] = {}
        self._vao_atlas: dict[tuple[str, bool], moderngl.VertexArray] = {}
        for nazwa, w in self.model.warstwy.items():
            trojkaty = ctx.buffer(w["trojkaty"].astype("u4").tobytes())
            for lewa in (False, True):
                poz, norm, kier = w["pozycje"].copy(), w["normalne"].copy(), w["kierunek"].copy()
                if lewa:
                    poz[:, 0] *= -1
                    norm[:, 0] *= -1
                    kier[:, 0] *= -1
                bufor = ctx.buffer(np.hstack([poz, norm, w["kosci"], w["wagi"], w["przedramie"], kier]).astype("f4").tobytes())
                self._vao[(nazwa, lewa)] = ctx.vertex_array(
                    self._anat,
                    [(bufor, "3f 3f 4f 4f 1f 12x", "in_poz", "in_norm", "in_kosci", "in_wagi", "in_przedramie")],
                    index_buffer=trojkaty,
                )
                self._vao_atlas[(nazwa, lewa)] = ctx.vertex_array(
                    self._atlas,
                    [(bufor, "3f 3f 4f 4f 1f 3f", "in_poz", "in_norm", "in_kosci", "in_wagi", "in_przedramie", "in_kier")],
                    index_buffer=trojkaty,
                )
        self._punkty_lewe = self.model.punkty * [-1.0, 1.0, 1.0]
        self._lokiec_lewy = self.model.lokiec * [-1.0, 1.0, 1.0]
        self._swiatlo: moderngl.Framebuffer | None = None   # bufor na zsumowane światło anatomii (rentgen)
        self._atlas_ms: moderngl.Framebuffer | None = None  # bufor z wygładzaniem krawędzi (anatomia)
        self._atlas_wynik: moderngl.Framebuffer | None = None

    def ustaw_obraz(self, dane: bytes, szer: int, wys: int, wyrownanie: int = 1) -> None:
        if self._tekstura is None or self._tekstura.size != (szer, wys):
            if self._tekstura is not None:
                self._tekstura.release()
            self._tekstura = self.ctx.texture((szer, wys), 3, alignment=wyrownanie)
            self._tekstura.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._tekstura.write(dane, alignment=wyrownanie)

    def _bufor_swiatla(self, w: int, h: int) -> moderngl.Framebuffer:
        if self._swiatlo is None or self._swiatlo.size != (w, h):
            if self._swiatlo is not None:
                self._swiatlo.color_attachments[0].release()
                self._swiatlo.release()
            tekstura = self.ctx.texture((w, h), 3, dtype="f2")   # zmiennoprzecinkowy: światło może być > 1
            tekstura.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._swiatlo = self.ctx.framebuffer(color_attachments=[tekstura])
        return self._swiatlo

    def _bufory_atlasu(self, w: int, h: int) -> tuple[moderngl.Framebuffer, moderngl.Framebuffer]:
        if self._atlas_ms is None or self._atlas_ms.size != (w, h):
            for fbo in (self._atlas_ms, self._atlas_wynik):
                if fbo is not None:
                    for zalacznik in (*fbo.color_attachments, fbo.depth_attachment):
                        if zalacznik is not None:
                            zalacznik.release()
                    fbo.release()
            probki = min(PROBKI_MSAA, self.ctx.max_samples)
            self._atlas_ms = self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer((w, h), 4, samples=probki)],
                depth_attachment=self.ctx.depth_renderbuffer((w, h), samples=probki))
            tekstura = self.ctx.texture((w, h), 4)
            tekstura.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._atlas_wynik = self.ctx.framebuffer(color_attachments=[tekstura])
        return self._atlas_ms, self._atlas_wynik

    def _macierze(self, d: Dlon, szer: int, wys: int, lustro: bool) -> tuple[bool, np.ndarray, tuple]:
        """Czy rysować odbity model, 21 macierzy (gotowe dla OpenGL) i zakres znikania przedramienia."""
        # Model to prawa ręka. Lewą (albo prawą w lustrzanym obrazie) rysujemy z odbitego modelu.
        lewa_geometrycznie = (d.strona == "Lewa") != lustro
        R = self._punkty_lewe if lewa_geometrycznie else self.model.punkty
        Re = self._lokiec_lewy if lewa_geometrycznie else self.model.lokiec
        P, Pe = punkty_na_obrazie(d, szer, wys)
        M = poza(P, R, Pe, Re).astype("f4")
        if Pe is None:
            zanik = ZANIK_BEZ_LOKCIA
        else:
            zanik = tuple(self.model.dlugosc_przedramienia + z for z in ZANIK_ZA_LOKCIEM)
        return lewa_geometrycznie, np.ascontiguousarray(M.transpose(0, 2, 1)), zanik

    def rysuj(self, cel: moderngl.Framebuffer, obszar: tuple[int, int, int, int], szer: int, wys: int,
              dlonie: list[Dlon], lustro: bool, warstwy: set[str], przyciemnienie: float,
              ekspozycja: float = 1.8, styl: str = "rentgen") -> None:
        """
        cel, obszar: bufor i prostokąt (x, y od dołu, szerokość, wysokość), w którym ma się pojawić obraz.
        szer, wys: rozmiar obrazu z kamery (w nim są współrzędne punktów).
        styl: "rentgen" (świecąca klisza) albo "anatomia" (pełne bryły jak w atlasie).
        """
        if styl == "anatomia":
            self._rysuj_atlas(cel, obszar, szer, wys, dlonie, lustro, warstwy, przyciemnienie)
            return
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

        # 1) Anatomia: światło wszystkich warstw sumowane w osobnym buforze.
        swiatlo = self._bufor_swiatla(obszar[2], obszar[3])
        swiatlo.use()
        ctx.viewport = (0, 0, obszar[2], obszar[3])
        swiatlo.clear(0.0, 0.0, 0.0, 0.0)
        if warstwy and dlonie:
            self._rysuj_anatomie(szer, wys, dlonie, lustro, warstwy)

        # 2) Złożenie: obraz z kamery (przyciemniony jak klisza) + ściśnięte światło anatomii.
        cel.use()
        ctx.viewport = obszar
        ctx.disable(moderngl.BLEND)
        if self._tekstura is not None:
            self._tekstura.use(0)
        swiatlo.color_attachments[0].use(1)
        self._tlo["obraz"].value = 0
        self._tlo["swiatlo"].value = 1
        self._tlo["jest_obraz"].value = int(self._tekstura is not None)
        self._tlo["przyciemnienie"].value = float(przyciemnienie)
        self._tlo["ekspozycja"].value = float(ekspozycja)
        self._tlo_vao.render(moderngl.TRIANGLE_STRIP)

    def _rysuj_anatomie(self, szer: int, wys: int, dlonie: list[Dlon], lustro: bool, warstwy: set[str]) -> None:
        ctx = self.ctx
        # Rzut: piksele obrazu -> współrzędne ekranu (y w dół), głębia tylko w granicach przycinania.
        rzut = np.array([[2.0 / szer, 0, 0, -1], [0, -2.0 / wys, 0, 1], [0, 0, -1.0 / (4 * szer), 0], [0, 0, 0, 1]],
                        dtype="f4")
        self._anat["rzut"].write(rzut.T.copy().tobytes())
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE      # warstwy sumują się jak na kliszy
        for d in dlonie:
            lewa_geometrycznie, M, zanik = self._macierze(d, szer, wys, lustro)
            self._anat["kosci"].write(M.tobytes())
            self._anat["zanik"].value = zanik
            for nazwa in WARSTWY:
                if nazwa not in warstwy:
                    continue
                barwa, jasnosc, srodek = KOLORY[nazwa]
                self._anat["barwa"].value = barwa
                self._anat["jasnosc"].value = jasnosc * d.widocznosc
                self._anat["srodek"].value = srodek
                self._vao[(nazwa, lewa_geometrycznie)].render(moderngl.TRIANGLES)
        ctx.disable(moderngl.BLEND)

    def _rysuj_atlas(self, cel: moderngl.Framebuffer, obszar: tuple[int, int, int, int], szer: int, wys: int,
                     dlonie: list[Dlon], lustro: bool, warstwy: set[str], przyciemnienie: float) -> None:
        """Styl „Anatomia”: nieprzezroczyste, oświetlone bryły z buforem głębi, potem złożenie z obrazem."""
        ctx = self.ctx
        ms, wynik = self._bufory_atlasu(obszar[2], obszar[3])
        ms.use()
        ctx.viewport = (0, 0, obszar[2], obszar[3])
        ms.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        if warstwy and dlonie:
            ctx.disable(moderngl.BLEND | moderngl.CULL_FACE)
            ctx.enable(moderngl.DEPTH_TEST)
            # Głębia z jednej kamery jest względna (każda dłoń ma własne zero), więc dwie dłonie
            # rozdzielamy: większa na obrazie jest bliżej i dostaje bliższy przedział głębi.
            rozmiary = [_rozmiar_dloni(np.asarray(d.punkty, dtype=float)[:, :2] * [szer, wys]) for d in dlonie]
            ranga = np.argsort(np.argsort(rozmiary))          # 0 = najmniejsza (najdalej)
            n = len(dlonie)
            for d, k in zip(dlonie, ranga):
                przesuniecie = 0.0 if n == 1 else 0.5 - float(k) / (n - 1)
                rzut = np.array([[2.0 / szer, 0, 0, -1], [0, -2.0 / wys, 0, 1],
                                 [0, 0, 1.0 / (8 * szer), przesuniecie], [0, 0, 0, 1]], dtype="f4")
                self._atlas["rzut"].write(rzut.T.copy().tobytes())
                lewa, M, zanik = self._macierze(d, szer, wys, lustro)
                self._atlas["kosci"].write(M.tobytes())
                self._atlas["zanik"].value = zanik
                self._atlas["widocznosc"].value = float(d.widocznosc)
                for nazwa in WARSTWY:
                    if nazwa in warstwy:
                        self._atlas["material"].value = MATERIALY[nazwa]
                        self._vao_atlas[(nazwa, lewa)].render(moderngl.TRIANGLES)
            ctx.disable(moderngl.DEPTH_TEST)
        ctx.copy_framebuffer(wynik, ms)          # uśrednienie próbek = wygładzone krawędzie

        cel.use()
        ctx.viewport = obszar
        if self._tekstura is not None:
            self._tekstura.use(0)
        wynik.color_attachments[0].use(1)
        self._atlas_tlo["obraz"].value = 0
        self._atlas_tlo["anatomia"].value = 1
        self._atlas_tlo["jest_obraz"].value = int(self._tekstura is not None)
        self._atlas_tlo["przyciemnienie"].value = float(przyciemnienie)
        self._atlas_tlo["piksel"].value = (1.0 / obszar[2], 1.0 / obszar[3])
        self._atlas_tlo_vao.render(moderngl.TRIANGLE_STRIP)
