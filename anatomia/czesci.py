"""
Części prawej dłoni z BodyParts3D (numery FMA ze spisu isa_parts_list_e.txt).

BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International.
"""

# Kości: (FMA, opis, punkt bliższy, punkt dalszy) — punkty to numery MediaPipe, między którymi leży kość.
KOSCI_PALCOW = [
    # kciuk
    ("FMA24464", "I kość śródręcza", 1, 2),
    ("FMA24450", "paliczek bliższy kciuka", 2, 3),
    ("FMA24459", "paliczek dalszy kciuka", 3, 4),
    # wskazujący
    ("FMA24466", "II kość śródręcza", 0, 5),
    ("FMA24451", "paliczek bliższy wskazującego", 5, 6),
    ("FMA24455", "paliczek środkowy wskazującego", 6, 7),
    ("FMA24460", "paliczek dalszy wskazującego", 7, 8),
    # środkowy
    ("FMA24468", "III kość śródręcza", 0, 9),
    ("FMA24452", "paliczek bliższy środkowego", 9, 10),
    ("FMA24456", "paliczek środkowy środkowego", 10, 11),
    ("FMA24461", "paliczek dalszy środkowego", 11, 12),
    # serdeczny
    ("FMA24470", "IV kość śródręcza", 0, 13),
    ("FMA24453", "paliczek bliższy serdecznego", 13, 14),
    ("FMA24457", "paliczek środkowy serdecznego", 14, 15),
    ("FMA24462", "paliczek dalszy serdecznego", 15, 16),
    # mały
    ("FMA24472", "V kość śródręcza", 0, 17),
    ("FMA24454", "paliczek bliższy małego", 17, 18),
    ("FMA24458", "paliczek środkowy małego", 18, 19),
    ("FMA24463", "paliczek dalszy małego", 19, 20),
]

KOSCI_NADGARSTKA = [
    ("FMA24435", "łódeczkowata"), ("FMA24437", "księżycowata"), ("FMA24439", "trójgraniasta"),
    ("FMA24441", "grochowata"), ("FMA24443", "czworoboczna większa"), ("FMA23725", "czworoboczna mniejsza"),
    ("FMA24446", "główkowata"), ("FMA24448", "haczykowata"),
]

# Kości przedramienia, całe (od nadgarstka do łokcia). Bez śledzenia łokcia widać tylko ich dolne końce.
KOSCI_PRZEDRAMIENIA = [("FMA23464", "kość promieniowa"), ("FMA23467", "kość łokciowa")]

# Pasmo tkanki łącznej nad kanałem nadgarstka (rysowane razem ze ścięgnami).
TROCZKI = [("FMA40120", "troczek zginaczy")]

# Pochewki ścięgien zginaczy na palcach — w BodyParts3D ich nie ma, więc anatomia/przygotuj.py
# buduje je wokół ścięgien. (punkty MediaPipe: nasada, staw podstawny, kolejne stawy…, czubek; ścięgna w środku)
POCHEWKI = [
    ((1, 2, 3, 4), ("FMA38482",)),                        # kciuk: zginacz długi kciuka
    ((0, 5, 6, 7, 8), ("FMA38470", "FMA38479")),          # wskazujący: zginacz powierzchowny i głęboki
    ((0, 9, 10, 11, 12), ("FMA38470", "FMA38479")),       # środkowy
    ((0, 13, 14, 15, 16), ("FMA38470", "FMA38479")),      # serdeczny
    ((0, 17, 18, 19, 20), ("FMA38470", "FMA38479")),      # mały
]

MIESNIE_DLONI = [
    ("FMA37386", "odwodziciel krótki kciuka"), ("FMA37388", "zginacz krótki kciuka"),
    ("FMA37390", "przeciwstawiacz kciuka"), ("FMA46121", "przywodziciel kciuka (głowa skośna)"),
    ("FMA46123", "przywodziciel kciuka (głowa poprzeczna)"),
    ("FMA37396", "odwodziciel palca małego"), ("FMA37398", "zginacz krótki palca małego"),
    ("FMA37400", "przeciwstawiacz palca małego"),
    ("FMA42398", "mięśnie glistowate"), ("FMA42402", "międzykostne dłoniowe"), ("FMA42404", "międzykostne grzbietowe"),
]

# Mięśnie przedramienia (od łokcia). Skrypt dzieli każdy na brzusiec (czerwony, warstwa „mięśnie”)
# i ścięgno (białe, warstwa „ścięgna”) według grubości. Opcje:
#   "brzusiec"      – cały jako mięsień (płaskie mięśnie, które inaczej wyszłyby na ścięgno),
#   "do_nadgarstka" – tylko część w przedramieniu.
MIESNIE_PRZEDRAMIENIA = [
    # zginacze palców i kciuka (ścięgna biegną do palców)
    ("FMA38470", "zginacz powierzchowny palców", ""), ("FMA38479", "zginacz głęboki palców", ""),
    ("FMA38482", "zginacz długi kciuka", ""),
    # Dłoniowy długi tylko w przedramieniu: w BodyParts3D zawiera rozcięgno dłoniowe, które zasłania
    # całe wnętrze dłoni; na rycinach w atlasach jest ono zdjęte, żeby było widać mięśnie.
    ("FMA38463", "dłoniowy długi", "do_nadgarstka"),
    # prostowniki palców i kciuka
    ("FMA38501", "prostownik palców", ""), ("FMA38525", "prostownik wskaziciela", ""),
    ("FMA38504", "prostownik palca małego", ""), ("FMA38522", "prostownik długi kciuka", ""),
    ("FMA38519", "prostownik krótki kciuka", ""), ("FMA38516", "odwodziciel długi kciuka", ""),
    # mięśnie nadgarstka: ich ścięgna kończą się na kościach nadgarstka i śródręcza
    ("FMA38460", "zginacz promieniowy nadgarstka", ""),
    ("FMA38617", "zginacz łokciowy nadgarstka (głowa ramienna)", ""),
    ("FMA38619", "zginacz łokciowy nadgarstka (głowa łokciowa)", ""),
    ("FMA38495", "prostownik promieniowy długi nadgarstka", ""),
    ("FMA38498", "prostownik promieniowy krótki nadgarstka", ""),
    ("FMA38507", "prostownik łokciowy nadgarstka", ""),
    # pozostałe mięśnie przedramienia
    ("FMA38486", "ramienno-promieniowy", ""),
    ("FMA38560", "nawrotny obły (głowa ramienna)", ""), ("FMA38562", "nawrotny obły (głowa łokciowa)", ""),
    ("FMA38454", "nawrotny czworoboczny", "brzusiec"), ("FMA38513", "odwracacz", "brzusiec"),
    ("FMA37705", "łokciowy", "brzusiec"),
]
