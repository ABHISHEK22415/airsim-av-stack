#!/usr/bin/env python3
"""Fly 3 drones in formation along the same path (AirSim must be running with the 3-drone settings.json).

    .venv/bin/python fly_swarm.py                    # figure-8 at 20/23/26 m
    .venv/bin/python fly_swarm.py --shape circle --size 50 --speed 6

Each drone's API coordinates are relative to its own spawn point, so giving all drones the same path
keeps them in a fixed formation (the spawn offsets in settings.json). Altitudes are staggered by 3 m
for extra safety.
"""
import argparse
import math
import time

import airsim

from fly_path import make_path

DRONES = {  # name: (spawn X, spawn Y) from settings.json, path colour
    "Drone1": ((0, 0), [1.0, 0.45, 0.1, 1.0]),
    "Drone2": ((-4, -5), [0.1, 0.6, 1.0, 1.0]),
    "Drone3": ((-4, 5), [0.2, 1.0, 0.3, 1.0]),
}
# FPV camera tilt per drone (degrees, negative = down): forward / 45 deg oblique / straight down (mapping)
CAM_PITCH = {"Drone1": 0, "Drone2": -45, "Drone3": -90}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["figure8", "circle", "square"], default="figure8")
    ap.add_argument("--alt", type=float, default=20.0, help="altitude of Drone1 (others +3 m, +6 m)")
    ap.add_argument("--size", type=float, default=50.0)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--laps", type=int, default=1)
    args = ap.parse_args()

    c = airsim.MultirotorClient()
    c.confirmConnection()
    names = list(DRONES)
    alts = {n: args.alt + 3 * i for i, n in enumerate(names)}

    for n in names:
        c.simSetCameraPose("front_center", airsim.Pose(airsim.Vector3r(0.5, 0, 0.1),
                           airsim.to_quaternion(math.radians(CAM_PITCH[n]), 0, 0)), vehicle_name=n)
        c.enableApiControl(True, n)
        c.armDisarm(True, n)
    try:
        print("Taking off (all 3)...")
        for f in [c.takeoffAsync(vehicle_name=n) for n in names]:
            f.join()
        for f in [c.moveToZAsync(-alts[n], 3, vehicle_name=n) for n in names]:
            f.join()

        c.simFlushPersistentMarkers()
        paths = {}
        for n in names:
            paths[n] = make_path(args.shape, alts[n], args.size)
            (sx, sy), col = DRONES[n]
            world = [airsim.Vector3r(p.x_val + sx, p.y_val + sy, p.z_val) for p in paths[n]]
            c.simPlotLineStrip([airsim.Vector3r(sx, sy, -alts[n])] + world, color_rgba=col,
                               thickness=6, is_persistent=True)

        for lap in range(1, args.laps + 1):
            print(f"Lap {lap}/{args.laps}")
            futs = [c.moveOnPathAsync(paths[n], args.speed, timeout_sec=600,
                                      drivetrain=airsim.DrivetrainType.ForwardOnly,
                                      yaw_mode=airsim.YawMode(False, 0), lookahead=-1,
                                      adaptive_lookahead=1, vehicle_name=n) for n in names]
            t0 = time.time()
            while True:
                line, done = [], 0
                for n in names:
                    p = c.getMultirotorState(vehicle_name=n).kinematics_estimated.position
                    line.append(f"{n}: ({p.x_val:5.1f},{p.y_val:5.1f}) {-p.z_val:4.1f}m")
                    last = paths[n][-1]
                    if time.time() - t0 > 3 and math.hypot(p.x_val - last.x_val, p.y_val - last.y_val) < 2.5:
                        done += 1
                print("\r  " + " | ".join(line), end="", flush=True)
                if done == len(names):
                    break
                time.sleep(0.5)
            for f in futs:
                f.join()
            print()

        print("Returning home and landing...")
        for f in [c.moveToPositionAsync(0, 0, -alts[n], 5, vehicle_name=n) for n in names]:
            f.join()
        for f in [c.landAsync(vehicle_name=n) for n in names]:
            f.join()
    except KeyboardInterrupt:
        print("\nInterrupted - landing all.")
        for f in [c.landAsync(vehicle_name=n) for n in names]:
            f.join()
    finally:
        for n in names:
            c.armDisarm(False, n)
            c.enableApiControl(False, n)
        print("Done.")


if __name__ == "__main__":
    main()
