"""Model ciała (Pose Landmarker) na zdjęciu testowym: gdzie są nadgarstki i łokcie, ile trwa analiza."""
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

BASE = Path(__file__).resolve().parents[1]
TESTY = Path(__file__).resolve().parent
model = sys.argv[1] if len(sys.argv) > 1 else "pose_landmarker_full.task"

opcje = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(BASE / "modele" / model)),
    running_mode=vision.RunningMode.VIDEO, num_poses=1)
det = vision.PoseLandmarker.create_from_options(opcje)
bgr = cv2.imread(str(TESTY / "dlonie_test.jpg"))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
obraz = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
czasy = []
for i in range(40):
    t = time.perf_counter()
    wynik = det.detect_for_video(obraz, i * 33)
    czasy.append((time.perf_counter() - t) * 1000)
print(f"czas: mediana {np.median(czasy[5:]):.1f} ms")
for p, sw in zip(wynik.pose_landmarks, wynik.pose_world_landmarks):
    for nazwa, (ra, lo, na) in {"lewa ręka": (11, 13, 15), "prawa ręka": (12, 14, 16)}.items():
        print(nazwa, {k: (round(p[i].x * W), round(p[i].y * H), round(p[i].visibility, 2), round(sw[i].z, 3))
                      for k, i in (("bark", ra), ("łokieć", lo), ("nadgarstek", na))})
    for i in (11, 12, 13, 14, 15, 16):
        cv2.circle(bgr, (int(p[i].x * W), int(p[i].y * H)), 6, (0, 255, 255), -1)
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):
        cv2.line(bgr, (int(p[a].x * W), int(p[a].y * H)), (int(p[b].x * W), int(p[b].y * H)), (0, 200, 255), 2)
cv2.imwrite(str(TESTY / "poza_test.png"), bgr)
det.close()
