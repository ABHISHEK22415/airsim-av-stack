"""Compute benchmark for the low-power goal: YOLO (threads / batch / input size) and lattice planner cost.
Run: .venv/bin/python bench_compute.py   (no simulator needed)"""
import sys, time, os, json
import numpy as np, cv2, torch
from ultralytics import YOLO
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
img = cv2.resize(cv2.imread("../docs/chase_view.jpg"), (640, 360))
res = {}
m = YOLO("yolov8n.pt")
for th in (8, 4, 1):
    torch.set_num_threads(th)
    for n, sz in ((1, 416), (6, 416), (2, 416), (6, 320)):
        batch = [img] * n
        m.predict(batch, imgsz=sz, verbose=False)
        t = time.time(); R = 5
        for _ in range(R): m.predict(batch, imgsz=sz, verbose=False)
        res[f"yolo threads={th} imgs={n} imgsz={sz}"] = round((time.time() - t) / R * 1000)
# lattice planner evaluation cost: 12 offsets x 45 samples vs N obstacle points, v2 (1 sweep) vs v3b (3 sweeps)
torch.set_num_threads(8)
for nobs in (200, 800):
    near = np.random.rand(nobs, 2) * 50
    for sweeps in (1, 3):
        pts = np.random.rand(45 * sweeps, 2) * 50
        t = time.time(); R = 50
        for _ in range(R):
            for off in range(12):
                np.sqrt(((pts[:, None, :] - near[None, :, :]) ** 2).sum(-1)).min(1)
        res[f"lattice 12 cands, {sweeps} sweep(s), {nobs} obstacle pts"] = round((time.time() - t) / R * 1000, 1)
print(json.dumps(res, indent=1))
