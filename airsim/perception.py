"""Camera perception for the AV stack (runs in its own process).

* 6-camera rig (Tesla-style): front narrow + wide, left/right B-pillar, rear-left/rear-right repeaters.
* RCCC sensor model: the RGB render is converted to a raw RCCC mosaic (2x2 cell = 1 Red-filtered pixel +
  3 Clear pixels), then "demosaiced" into a full-res CLEAR (luminance) image and a half-res RED plane.
  All perception uses only these two signals - no RGB colour is available, exactly like an RCCC camera.
* Object detection: YOLOv8n on the CLEAR image (person, bicycle, car, motorcycle, bus, truck).
* Traffic lights: map-guided ROI (HD-map lamp position projected into the camera) + RED/CLEAR ratio
  classifier (red lamp: high R/C, yellow: medium R/C and bright, green: low R/C).
* "Vector space": every detection's ground contact point is ray-cast onto the road plane -> world x,y.
"""
import math
import os
import time

import cv2
import numpy as np

CAMS = ["cam_front_narrow", "cam_front_wide", "cam_left", "cam_right", "cam_rear_left", "cam_rear_right"]
FRONT = CAMS[:2]
LABEL = {"cam_front_narrow": "FRONT NARROW 50", "cam_front_wide": "FRONT WIDE 120", "cam_left": "LEFT PILLAR",
         "cam_right": "RIGHT PILLAR", "cam_rear_left": "REAR-LEFT REPEATER", "cam_rear_right": "REAR-RIGHT REPEATER"}
FOV = {"cam_front_narrow": 50, "cam_front_wide": 120, "cam_left": 100, "cam_right": 100,
       "cam_rear_left": 90, "cam_rear_right": 90}
YOLO_CLASSES = [0, 1, 2, 3, 5, 7]   # person, bicycle, car, motorcycle, bus, truck
HERE = os.path.dirname(os.path.abspath(__file__))
# camera calibration from calibrate_cams.py (focal length per camera + vehicle/world vertical frame offset)
try:
    import json
    CALIB = json.load(open(os.path.join(HERE, "cam_calib.json")))
except Exception:
    CALIB = {}
DZ = float(np.mean([v["dz"] for v in CALIB.values()])) if CALIB else -1.1


def focal(cam, W=640):
    if cam in CALIB:
        return CALIB[cam]["f"]
    return (W / 2) / math.tan(math.radians(FOV[cam]) / 2)


# ---------------- RCCC sensor model ----------------
def rccc_mosaic(bgr, rng=np.random.default_rng(0)):
    """RGB render -> raw RCCC mosaic (uint8). Pattern per 2x2 cell: [R C / C C]."""
    f = bgr.astype(np.float32)
    clear = 0.30 * f[..., 2] + 0.45 * f[..., 1] + 0.25 * f[..., 0]   # clear = broad panchromatic response
    raw = clear.copy()
    raw[0::2, 0::2] = f[0::2, 0::2, 2]                                  # red-filtered photosites
    raw += rng.normal(0, 1.5, raw.shape)                                # read noise
    return np.clip(raw, 0, 255).astype(np.uint8)


def rccc_demosaic(raw):
    """raw RCCC -> (clear full-res uint8, red half-res float upsampled to full-res)."""
    c = raw.astype(np.float32)
    clear = c.copy()
    # fill red sites with the mean of their 4 clear neighbours
    pad = np.pad(c, 1, mode="edge")
    nb = (pad[0:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, 0:-2] + pad[1:-1, 2:]) / 4.0
    clear[0::2, 0::2] = nb[0::2, 0::2]
    red = cv2.resize(c[0::2, 0::2], (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
    return clear.astype(np.uint8), red


def rccc_display(clear, red):
    """Grey clear image with red-dominant pixels tinted red (what an RCCC camera can 'see')."""
    g = cv2.cvtColor(clear, cv2.COLOR_GRAY2BGR).astype(np.float32)
    ratio = red / (clear.astype(np.float32) + 8.0)
    m = np.clip((ratio - 1.35) * 3.0, 0, 1)[..., None]
    tint = np.zeros_like(g)
    tint[..., 2] = 255
    return (g * (1 - 0.7 * m) + tint * 0.7 * m).astype(np.uint8)


# ---------------- geometry ----------------
def quat_to_R(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(pw, cam_pos, R, W, H, f):
    """World point -> (u, v, depth) or None. Camera body frame: x fwd, y right, z down."""
    pc = R.T @ (np.asarray(pw, float) + np.array([0, 0, DZ]) - cam_pos)
    if pc[0] < 1.0:
        return None
    return W / 2 + f * pc[1] / pc[0], H / 2 + f * pc[2] / pc[0], pc[0]


def ground_point(u, v, cam_pos, R, W, H, f):
    """Ray-cast an image pixel onto the road plane (world z = 0) -> world (x, y)."""
    ray = R @ np.array([1.0, (u - W / 2) / f, (v - H / 2) / f])
    if ray[2] <= 1e-3:
        return None
    t = (DZ - cam_pos[2]) / ray[2]
    if t <= 0 or t > 80:
        return None
    p = cam_pos + t * ray
    return float(p[0]), float(p[1])


def classify_lamp(clear, red, u, v, rad):
    """RCCC traffic-light state from the R/C ratio inside the lamp disc. Returns (state, ratio, clear_mean)."""
    H, W = clear.shape
    r = max(2, int(rad * 1.0))
    x0, x1, y0, y1 = int(u) - r, int(u) + r + 1, int(v) - r, int(v) + r + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None, 0, 0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
    C = clear[y0:y1, x0:x1].astype(np.float32)[disc]
    Rr = red[y0:y1, x0:x1][disc]
    if C.size < 4:
        return None, 0, 0
    ratio = float(np.percentile(Rr / (C + 8.0), 75))   # lamp pixels dominate the upper quantile
    cm = float(np.median(C))
    if ratio < 1.0:
        st = "G"
    elif ratio > 1.45:
        st = "R"
    else:
        st = "Y"
    return st, ratio, cm


# ---------------- process ----------------
def perception_proc(lamp_arr, out_q, stop_evt):
    import airsim
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model = YOLO(os.path.join(HERE, "yolov8n.pt"))
    c = airsim.CarClient()
    c.confirmConnection()
    for n in CAMS:
        c.simSetCameraFov(n, FOV[n])
    last = {n: (np.zeros((360, 640, 3), np.uint8), []) for n in CAMS}
    k = 0
    while not stop_evt.is_set():
        names = CAMS if k % 3 == 0 else FRONT
        k += 1
        t0 = time.time()
        try:
            rs = c.simGetImages([airsim.ImageRequest(n, airsim.ImageType.Scene, False, False) for n in names])
        except Exception:
            time.sleep(0.2)
            continue
        frames = {}
        for n, r in zip(names, rs):
            if r.width == 0:
                continue
            bgr = np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3)
            clear, red = rccc_demosaic(rccc_mosaic(bgr))
            cp = np.array([r.camera_position.x_val, r.camera_position.y_val, r.camera_position.z_val])
            frames[n] = (clear, red, cp, quat_to_R(r.camera_orientation), r.width, r.height)
        if not frames:
            continue
        res = model.predict([cv2.cvtColor(f[0], cv2.COLOR_GRAY2BGR) for f in frames.values()], imgsz=416,
                            conf=0.35, classes=YOLO_CLASSES, verbose=False)
        objs = []
        for (n, (clear, red, cp, R, W, H)), rr in zip(frames.items(), res):
            disp = rccc_display(clear, red)
            dets = []
            for b in rr.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                name = model.names[int(b.cls)]
                gp = ground_point((x1 + x2) / 2, y2, cp, R, W, H, focal(n, W))
                dets.append((name, float(b.conf), x1, y1, x2, y2, gp))
                if gp:
                    objs.append((name, gp[0], gp[1], n))
                col = (0, 200, 255) if name in ("person", "bicycle") else (255, 160, 0)
                cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
                cv2.putText(disp, f"{name} {b.conf.item():.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 1, cv2.LINE_AA)
            last[n] = (disp, dets)
        # traffic light (front cameras, map-guided ROI)
        tl = (None, 0.0, 0.0, None)
        lx, ly, lz, valid = lamp_arr[:]
        if valid > 0.5:
            for n in ["cam_front_narrow"]:   # lights are read by the narrow main camera only (wide = too few px)
                if n not in frames:
                    continue
                clear, red, cp, R, W, H = frames[n]
                f = focal(n, W)
                pr = project((lx, ly, lz), cp, R, W, H, f)
                if pr is None or not (8 < pr[0] < W - 8 and 8 < pr[1] < H - 8):
                    continue
                rad = max(2.5, 0.35 * f / pr[2])
                st, ratio, cm = classify_lamp(clear, red, pr[0], pr[1], rad)
                disp = last[n][0]
                colr = {"R": (0, 0, 255), "Y": (0, 220, 255), "G": (0, 255, 0), None: (200, 200, 200)}[st]
                rr_ = int(rad * 2.5) + 6
                cv2.rectangle(disp, (int(pr[0]) - rr_, int(pr[1]) - rr_), (int(pr[0]) + rr_, int(pr[1]) + rr_), colr, 2)
                cv2.putText(disp, f"TL {st} R/C={ratio:.2f}", (int(pr[0]) - rr_, int(pr[1]) - rr_ - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colr, 1, cv2.LINE_AA)
                tl = (st, ratio, cm, n)
                break
        # 3x2 camera grid like a Tesla debug view
        tiles = []
        for n in CAMS:
            im = cv2.resize(last[n][0], (400, 225))
            cv2.rectangle(im, (0, 0), (400, 20), (0, 0, 0), -1)
            cv2.putText(im, LABEL[n] + "  [RCCC]", (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(im)
        grid = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6])])
        counts = {}
        for n in CAMS:
            for d in last[n][1]:
                counts[d[0]] = counts.get(d[0], 0) + 1
        msg = dict(t=t0, tl=tl, objs=objs, counts=counts, fps=1.0 / max(1e-3, time.time() - t0),
                   front=[(d[0], d[2], d[3], d[4], d[5], cam) for cam in FRONT for d in last[cam][1]],
                   grid=cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        try:
            while True:
                out_q.get_nowait()
        except Exception:
            pass
        out_q.put(msg)
