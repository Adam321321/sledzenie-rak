# Śledzenie dłoni

Aplikacja na Windowsa: wybierasz kamerę, a ona rysuje na podglądzie szkielet dłoni, czyli
nadgarstek i wszystkie stawy palców (21 punktów na dłoń, do dwóch dłoni naraz).

| Do czego | Biblioteka |
|---|---|
| Śledzenie dłoni | **Google MediaPipe** – Hand Landmarker |
| Łokcie (kierunek przedramienia) | **Google MediaPipe** – Pose Landmarker (model ciała) |
| Okno | **Qt** (PySide6) |
| Obraz z kamery | **OpenCV** |
| Lista kamer z nazwami | Windows DirectShow (przez `comtypes`) |

## Uruchamianie

Kliknij dwa razy skrót **„Śledzenie dłoni”** na pulpicie (albo `start.bat` w tym folderze).

1. Wybierz kamerę z listy. **OBS Virtual Camera** to twój iPhone z aplikacji Kamerka (najpierw włącz Kamerkę i nadawanie).
2. Kliknij **Start**.

## Opcje w oknie

| Opcja | Co robi |
|---|---|
| **↻** | odświeża listę kamer (np. po podłączeniu nowej) |
| **Jakość** | rozdzielczość obrazu z kamery; *Auto* bierze najwyższą obsługiwaną (do 1080p) |
| **1 dłoń / 2 dłonie** | ile dłoni śledzić naraz; jedna liczy się dwa razy szybciej i dokładniej śledzi szybki ruch |
| **Wygładzanie** | tłumi drżenie punktów (Wył. / Słabe / Średnie / Mocne); zmienia się w locie, „Wył.” = surowy wynik modelu do porównania |
| **Widok** | *Szkielet* (kolorowe punkty), *Anatomia* (kości, mięśnie i ścięgna w 3D jak na rycinie z atlasu), *Rentgen* (to samo jako świecąca klisza) albo *Anatomia / Rentgen + szkielet* |
| **Kości / Mięśnie / Ścięgna** | które warstwy anatomii pokazać (np. same kości albo kości z mięśniami) |
| **Lustro** | odbicie lustrzane jak w selfie: ruszasz prawą ręką, rusza się ta po prawej stronie ekranu |
| **Numery punktów** | pokazuje numery 0–20 przy stawach (przydatne przy programowaniu gestów) |
| **Łokieć** | śledzi łokieć modelem ciała, więc przedramię w widokach Anatomia/Rentgen idzie od nadgarstka do łokcia (także przy zgiętym nadgarstku); w widoku Szkielet pojawia się przerywana linia do łokcia |

Na dole okna widać: ile dłoni wykryto (i która to która), ile klatek na sekundę idzie, ile trwa
analiza jednej klatki i jakie jest opóźnienie od kamery do ekranu.

Ustawienia (wybrana kamera, opcje, rozmiar okna) zapamiętują się same.

## Numery punktów

```
0  nadgarstek
1–4    kciuk        (4 = czubek)
5–8    wskazujący   (8 = czubek)
9–12   środkowy     (12 = czubek)
13–16  serdeczny    (16 = czubek)
17–20  mały         (20 = czubek)
```

Kolory na podglądzie: kciuk pomarańczowy, wskazujący żółty, środkowy zielony, serdeczny
błękitny, mały fioletowy.

## Wydajność

Na tym komputerze (Ryzen 7 2700) analiza jednej dłoni trwa ok. 27 ms, a dwóch ok. 51 ms.
MediaPipe na Windowsie nie korzysta z karty graficznej, tylko z procesora. Dlatego aplikacja
uruchamia **dwa modele równolegle**, które biorą na zmianę co drugą klatkę. Daje to ok. 36 kl./s
przy dwóch dłoniach, więc kamera 30 kl./s idzie płynnie. Liczbę modeli ustawia stała
`LICZBA_WATKOW_ANALIZY` w `aplikacja.py`. Przy **jednej dłoni** wystarcza jeden model, który
widzi każdą klatkę: śledzi wtedy dokładniej w ruchu, a opóźnienie spada z ok. 75 do ok. 38 ms.

Model ciała (łokcie) jest wolniejszy: ok. 60–70 ms na klatkę. Dlatego liczy we własnym wątku, w swoim
tempie (ok. 15 razy na sekundę), a do każdej klatki z dłońmi dokładany jest jego najnowszy wynik.
Śledzenie dłoni zostaje przy tej samej płynności (na zdjęciu testowym 26 kl./s bez łokci, 25 z łokciami).
Przedramię i tak zaczyna się w nadgarstku z modelu dłoni, aktualizowanym co klatkę.

## Stabilność

Surowy wynik MediaPipe lekko drży, bo każda klatka z kamery jest trochę inna (szum, kompresja,
światło). Moduł `stabilizacja.py` robi trzy rzeczy:

1. **Wygładzanie filtrem One Euro** (Casiez i in., CHI 2012). Mocno tłumi drżenie, gdy dłoń stoi,
   i sam się rozluźnia, gdy dłoń szybko się rusza, więc opóźnienie zostaje małe.
2. **Pilnowanie, która dłoń jest która.** Dłonie są dopasowywane między klatkami po położeniu,
   a podpis Lewa/Prawa ustalany jest z ostatnich ~0,4 s, więc nie przeskakuje.
3. **Ochrona przed miganiem.** Nowa dłoń pojawia się po 2 klatkach z rzędu, a zgubiona na moment
   przygasa przez ~0,15 s zamiast od razu znikać.

Pomiar: zdjęcie dłoni z dodanym szumem jak z kamerki i kompresją JPEG, 150 klatek. Drganie to
średni ruch punktu między klatkami, gdy dłoń stoi. Odstawanie to średnia odległość szkieletu
od prawdziwego położenia, gdy obraz szybko jeździ w lewo i prawo (±160 px, ok. 670 px/s).

| Wygładzanie | Drganie w bezruchu | Odstawanie w ruchu (2 dłonie) | Odstawanie w ruchu (1 dłoń) |
|---|---|---|---|
| Wył. | 0,61 px | 2,6 px | 1,1 px |
| Słabe | 0,14 px | 3,2 px | 1,5 px |
| **Średnie** (domyślne) | **0,07 px** | 4,6 px | 2,8 px |
| Mocne | 0,04 px | 6,3 px | 4,4 px |

Ustawienia filtra (`POZIOMY` w `stabilizacja.py`) dobrano, sprawdzając 49 kombinacji: dla
każdego poziomu spokoju wybrano tę z najmniejszym odstawaniem w ruchu. Skrypty pomiarowe są
w folderze `pomiary/` (`pomiar_stabilnosci.py <zdjęcie z dłońmi> [plik_wyników.pkl]`, potem
`strojenie.py <plik_wyników.pkl>`).

**Najwięcej i tak daje światło.** W słabym świetle kamera wydłuża czas naświetlania, palce się
rozmazują i żaden filtr tego nie naprawi. Pomaga też gładkie tło i duża dłoń w kadrze.

## Anatomia i rentgen 3D

Prawdziwy, trójwymiarowy model anatomiczny dłoni dopasowany na żywo do twojej dłoni. Nie pokazuje
twoich prawdziwych kości, tylko przeciętną dłoń ułożoną tak jak twoja (poglądowo).

- **Model:** BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0
  International:
  - kości: 27 kości dłoni (paliczki, śródręcze, nadgarstek) oraz kość promieniowa i łokciowa,
  - mięśnie dłoni: kłąb kciuka (odwodziciel krótki, zginacz krótki, przeciwstawiacz, przywodziciel),
    kłąb palca małego, glistowate, międzykostne dłoniowe i grzbietowe,
  - 22 mięśnie przedramienia (zginacze i prostowniki palców, kciuka i nadgarstka, ramienno-promieniowy,
    nawrotne, odwracacz, łokciowy). Skrypt dzieli każdy z nich według grubości na czerwony brzusiec
    i białe ścięgno,
  - troczek zginaczy nad kanałem nadgarstka.
  - Pochewki ścięgien zginaczy na palcach (z pogrubieniami w miejscu więzadeł obrączkowatych A1–A5)
    nie istnieją w BodyParts3D, więc skrypt buduje je sam wokół ścięgien.
  - Pominięty jest mięsień dłoniowy długi: w BodyParts3D zawiera rozcięgno dłoniowe, które zasłania
    całe wnętrze dłoni (na rycinach w atlasach też jest zdjęte).
- **Dopasowanie:** ze wzorcowej dłoni znamy położenie tych samych 21 punktów co w MediaPipe. Każda
  kość jest obracana i rozciągana tak, żeby leżała między „swoimi” punktami na twojej dłoni.
  Mięśnie i ścięgna wyginają się razem z kilkoma kośćmi naraz. Lewa dłoń to lustrzane odbicie prawej.
- **Przedramię:** model dłoni nie widzi przedramienia, więc kierunek bierzemy z modelu ciała (opcja
  **Łokieć**): przedramię idzie od nadgarstka do łokcia i obraca się razem z dłonią (oś w poprzek dłoni).
  Przy nadgarstku tkanki płynnie przechodzą z dłoni na przedramię, więc zgięcie nadgarstka jest widoczne.
  Łokieć jest wygładzany i przytrzymywany przez 0,5 s, gdy model ciała go chwilowo zgubi. Gdy łokcia nie
  widać (poza kadrem, opcja wyłączona), przedramię trzyma się dłoni i łagodnie zanika za nadgarstkiem.
  Model ciała szuka najpierw głowy i tułowia, więc łokieć znajdzie, gdy w kadrze jest też osoba, a nie
  tylko ręka z bliska. Skrzyżowanie kości promieniowej z łokciową przy obracaniu dłoni nie jest pokazane:
  całe przedramię obraca się razem z dłonią (poglądowo).
- **Widok Anatomia:** pełne, oświetlone bryły, które zasłaniają się jak w rzeczywistości. Gdy dłoń jest
  zwrócona wnętrzem do kamery, widać ścięgna zginaczy i mięśnie kłębów, a od grzbietu prostowniki
  i międzykostne grzbietowe. Mięśnie mają rysunek włókien, a ścięgna satynowy połysk. Krawędzie są
  wygładzane (4× MSAA), a pod modelem leży miękki cień.
- **Widok Rentgen:** jak klisza RTG: obraz przyciemniony na niebiesko, kości świecą mocniej na krawędziach,
  a warstwy sumują się, więc kości widać przez mięśnie. Jasność jest „ściskana” (tone mapping),
  żeby środek dłoni się nie przepalał.
- **Wydajność:** rysuje karta graficzna (biblioteka moderngl, osobny kontekst OpenGL), klatka trafia
  do podglądu. Na GTX 1660 Super nie zmienia liczby klatek na sekundę (ok. 27 kl./s na zdjęciu testowym).
- **Ograniczenia:** głębia z jednej kamery jest przybliżona, a model to przeciętna dłoń, więc
  nakładka może lekko „pływać”, zwłaszcza gdy dłoń jest bokiem albo palce celują w kamerę.
- Jeśli karta graficzna albo OpenGL nie zadziałają, opcja Rentgen się wyłączy, a szkielet działa dalej.

Model przygotowuje skrypt `anatomia/przygotuj.py` (wynik: `anatomia/dlon.npz`, 7,1 MB, ok. 1 min). Uruchamia
się go tylko przy zmianie części: lista jest w `anatomia/czesci.py`. Oryginalne archiwum BodyParts3D
(136 MB, `isa_BP3D_4.0_obj_99.zip`) trzeba umieścić w `anatomia/zrodlo/`. Nie ma go w repozytorium, bo
przekracza limit GitHuba. Aplikacja go nie potrzebuje, jest tylko do ponownego przygotowania modelu.

## Gdy coś nie działa

| Problem | Rozwiązanie |
|---|---|
| „Nie udało się otworzyć …” | Kamera jest wyłączona albo używa jej inna aplikacja. Kamery „Meta Quest” działają tylko, gdy gogle są podłączone. |
| OBS Virtual Camera pokazuje logo OBS albo czarny obraz | Włącz Kamerkę (`start.bat` w folderze kamerka-iphone) i kliknij Start na iPhonie. |
| Podgląd jest rozpikselowany | Ustaw **Jakość: Auto** albo **1080p**. Kamera OBS oddaje dokładnie taki rozmiar, o jaki się ją poprosi, i sama go skaluje bez wygładzania. Rozdzielczość, którą faktycznie dostajesz, widać na dole okna. |
| Przedramię nie idzie do łokcia („łokcie: 0” na dole okna) | Zaznacz **Łokieć**. Łokieć musi być w kadrze, a model ciała potrzebuje też głowy i tułowia (sama ręka z bliska nie wystarczy). |
| Lewa i prawa dłoń są zamienione | Sprawdź opcję **Lustro**. Podpis zawsze mówi, która to dłoń osoby w kadrze. |
| Aplikacja się nie włącza | Zajrzyj do pliku `aplikacja.log` w tym folderze. |

## Pliki

- `aplikacja.py` – okno, podgląd i rysowanie szkieletu.
- `dlonie.py` – samo śledzenie (MediaPipe), bez okna; da się go użyć gdzie indziej, np. do wysyłania pozycji palców do Unity.
- `stabilizacja.py` – wygładzanie, pilnowanie tożsamości dłoni i ochrona przed miganiem; też bez okna.
- `lokcie.py` – łokcie z modelu ciała (MediaPipe Pose): wykrywanie, przypisanie do dłoni, wygładzanie; też bez okna.
- `rentgen.py` – anatomia i rentgen 3D: dopasowanie modelu anatomicznego do punktów i rysowanie (OpenGL).
- `anatomia/` – model anatomiczny dłoni (`dlon.npz`), lista części i skrypt, który go przygotowuje.
- `testy/` – skrypty testowe i zdjęcie testowe (`test_anatomia.py` rysuje wzorcową dłoń, przedramię
  i zdjęcie w widoku Anatomia do PNG, `test_rentgen.py` to samo dla rentgena, `test_lokcie.py` sprawdza
  łączenie łokci z dłońmi).
- Kopia wersji sprzed rentgena („podstawka”): `A:\sledzenie-dloni-podstawa` (uruchamia się jej własnym `start.bat`).
- `kamery.py` – lista kamer z nazwami.
- `modele/hand_landmarker.task` – model dłoni od Google (7,8 MB, pobierany automatycznie, jeśli go brakuje).
- `modele/pose_landmarker_full.task` – model ciała od Google (9,4 MB, jw.).
- `instaluj.bat` – instaluje wszystko od nowa (np. na innym komputerze).
- Test bez kamery: `start.bat --plik zdjęcie_albo_film.mp4`.
