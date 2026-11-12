#!/usr/bin/env python3
"""Physical traffic lights for AirSimNH: a pole + lamp (sphere) for every approach of every signalised
junction. The lamp's texture is switched RED / YELLOW / GREEN so cameras can actually see it.
Timing comes from drive_to.light() (wall-clock based), so every process agrees on the state.

    .venv/bin/python signals.py        # until Ctrl+C (removes the lights on exit)
"""
import os
import signal
import time

import airsim
import cv2
import numpy as np

from drive_to import ADJ, SIGNALS, STOP_BACK, add, light, right, unit

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = {}
for k, bgr in {"R": (0, 0, 255), "Y": (0, 210, 255), "G": (0, 255, 0)}.items():
    TEX[k] = os.path.join(HERE, f"tex_{k}.png")
    img = np.zeros((64, 64, 3), np.uint8)
    img[:] = bgr
    cv2.imwrite(TEX[k], img)


def lamp_pos(node, d):
    """World position of the lamp for traffic approaching `node` in direction `d`: far-right corner of the
    junction (like real signals), so it is visible from the stop line."""
    p = add(add(node, d, 10.0), right(d), 6.0)
    return (p[0], p[1], -4.6)


def approaches():
    for n in SIGNALS:
        for m in ADJ[n]:
            d = unit(m, n)
            yield n, d, ("x" if abs(d[0]) > 0.5 else "y")


def _term(*_):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, _term)


def main():
    c = airsim.CarClient()
    c.confirmConnection()
    lamps, spawned = [], []
    for i, (n, d, axis) in enumerate(approaches()):
        x, y, z = lamp_pos(n, d)
        pole = c.simSpawnObject(f"sigpole_{i}", "Cylinder", airsim.Pose(airsim.Vector3r(x, y, -2.15), airsim.Quaternionr()),
                                airsim.Vector3r(0.15, 0.15, 4.3), False, False)
        lamp = c.simSpawnObject(f"siglamp_{i}", "Sphere", airsim.Pose(airsim.Vector3r(x, y, z), airsim.Quaternionr()),
                                airsim.Vector3r(0.7, 0.7, 0.7), False, False)
        spawned += [pole, lamp]
        lamps.append([lamp, n, axis, None])
    print(f"Spawned {len(lamps)} traffic lights at {len(SIGNALS)} junctions.", flush=True)
    try:
        while True:
            now = time.time()
            for L in lamps:
                st = light(L[1], L[2], now)
                if st != L[3]:
                    c.simSetObjectMaterialFromTexture(L[0], TEX[st])
                    L[3] = st
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for n in spawned:
            c.simDestroyObject(n)
        print("Removed traffic lights.")


if __name__ == "__main__":
    main()
