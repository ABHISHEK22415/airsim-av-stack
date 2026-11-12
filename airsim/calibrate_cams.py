#!/usr/bin/env python3
"""Camera intrinsic/extrinsic calibration against the sim (like a factory camera calibration).
Spawns red target spheres at known world positions in front of the car (car must be at its start pose),
detects them in each front camera and least-squares fits focal length f and the vehicle/world frame
z-offset dz.  Writes cam_calib.json used by perception.py.
"""
import json
import math
import os
import time

import airsim
import numpy as np

import signals
from perception import FOV, FRONT, quat_to_R

PRIOR_F = {"cam_front_narrow": 566.0, "cam_front_wide": 174.0}  # rough first-pass estimate
TARGETS = [(12, 3.0, -3.5), (15, -3.0, -1.5), (18, 4.0, -4.6), (20, 0.0, -1.0), (25, -5.0, -3.0),
           (30, 6.0, -4.6), (14, -1.5, -4.6), (22, 2.5, -2.0)]


def main():
    c = airsim.CarClient()
    c.confirmConnection()
    for cam in FRONT:
        c.simSetCameraFov(cam, FOV[cam])
    calib = {}
    obs = {cam: [] for cam in FRONT}
    for i, p in enumerate(TARGETS):
        n = c.simSpawnObject(f"calib_{i}", "Sphere", airsim.Pose(airsim.Vector3r(*p), airsim.Quaternionr()),
                             airsim.Vector3r(0.8, 0.8, 0.8), False, False)
        c.simSetObjectMaterialFromTexture(n, signals.TEX["R"])
        time.sleep(1.0)
        rs = c.simGetImages([airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False) for cam in FRONT])
        for cam, r in zip(FRONT, rs):
            bgr = np.frombuffer(r.image_data_uint8, np.uint8).reshape(r.height, r.width, 3).astype(int)
            cp = np.array([r.camera_position.x_val, r.camera_position.y_val, r.camera_position.z_val])
            pc = quat_to_R(r.camera_orientation).T @ (np.array(p) - cp)
            f0 = PRIOR_F[cam]
            u0, v0 = r.width / 2 + f0 * pc[1] / pc[0], r.height / 2 + f0 * (pc[2] - 1.0) / pc[0]
            mask = (bgr[..., 2] - np.maximum(bgr[..., 1], bgr[..., 0])) > 90
            yy, xx = np.mgrid[0:r.height, 0:r.width]
            mask &= (np.abs(xx - u0) < 45) & (np.abs(yy - v0) < 45)
            ys, xs = np.nonzero(mask)
            if len(xs) < 3:
                continue
            obs[cam].append((pc, xs.mean() - r.width / 2, ys.mean() - r.height / 2))
        c.simDestroyObject(n)
        time.sleep(0.3)
    for cam in FRONT:
        o = obs[cam]
        # u = f*y/x  -> f by least squares;  v = f*(z+dz)/x -> dz
        A = np.array([pc[1] / pc[0] for pc, u, v in o])
        U = np.array([u for pc, u, v in o])
        f = float(A @ U / (A @ A))
        dz = float(np.mean([v * pc[0] / f - pc[2] for pc, u, v in o]))
        res = np.sqrt(np.mean([(f * pc[1] / pc[0] - u) ** 2 + (f * (pc[2] + dz) / pc[0] - v) ** 2 for pc, u, v in o]))
        fov_eff = math.degrees(2 * math.atan(320 / f))
        calib[cam] = dict(f=f, dz=dz, fov_eff=fov_eff, rms_px=float(res), n=len(o))
        print(f"{cam}: f={f:.1f}px (effective FOV {fov_eff:.1f} deg, set {FOV[cam]}), dz={dz:.2f} m, "
              f"reprojection RMS {res:.1f}px from {len(o)} targets")
    json.dump(calib, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cam_calib.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
