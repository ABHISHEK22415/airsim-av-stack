#!/usr/bin/env python3
"""Fly the AirSim drone along a waypoint path.

Usage (simulator must already be running, see run_sim.sh):
    .venv/bin/python fly_path.py                      # figure-8, 1 lap
    .venv/bin/python fly_path.py --shape circle --laps 2 --alt 30 --speed 8

AirSim uses NED coordinates: x = north, y = east, z = DOWN (so altitude 25 m is z = -25).
"""
import argparse
import math
import time

import airsim


def make_path(shape: str, alt: float, size: float):
    z = -abs(alt)
    pts = []
    if shape == "square":
        for x, y in [(size, 0), (size, size), (0, size), (0, 0)]:
            pts.append((x, y))
    elif shape == "circle":
        for i in range(1, 37):
            a = 2 * math.pi * i / 36
            pts.append((size / 2 * math.sin(a), size / 2 * (1 - math.cos(a))))
    else:  # figure8 (lemniscate starting and ending at home)
        for i in range(1, 49):
            a = 2 * math.pi * i / 48
            pts.append((size * math.sin(a), size / 2 * math.sin(2 * a)))
    return [airsim.Vector3r(x, y, z) for x, y in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["figure8", "circle", "square"], default="figure8")
    ap.add_argument("--alt", type=float, default=25.0, help="altitude in metres")
    ap.add_argument("--size", type=float, default=60.0, help="path size in metres")
    ap.add_argument("--speed", type=float, default=7.0, help="m/s")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--vehicle", default="Drone1")
    args = ap.parse_args()

    c = airsim.MultirotorClient()
    c.confirmConnection()
    v = args.vehicle
    c.enableApiControl(True, v)
    c.armDisarm(True, v)

    try:
        print("Taking off...")
        c.takeoffAsync(vehicle_name=v).join()
        print(f"Climbing to {args.alt} m")
        c.moveToZAsync(-abs(args.alt), 3, vehicle_name=v).join()

        path = make_path(args.shape, args.alt, args.size)
        # draw the path in the sim (persistent orange line)
        c.simFlushPersistentMarkers()
        c.simPlotLineStrip([airsim.Vector3r(0, 0, -abs(args.alt))] + path,
                           color_rgba=[1.0, 0.45, 0.1, 1.0], thickness=8, is_persistent=True)

        for lap in range(1, args.laps + 1):
            print(f"Lap {lap}/{args.laps}: {args.shape}, {len(path)} waypoints @ {args.speed} m/s")
            f = c.moveOnPathAsync(path, args.speed, timeout_sec=600,
                                  drivetrain=airsim.DrivetrainType.ForwardOnly,
                                  yaw_mode=airsim.YawMode(False, 0),
                                  lookahead=-1, adaptive_lookahead=1, vehicle_name=v)
            # print simple telemetry while flying
            t0 = time.time()
            while True:
                s = c.getMultirotorState(vehicle_name=v).kinematics_estimated
                p, vel = s.position, s.linear_velocity
                spd = math.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2)
                print(f"\r  pos=({p.x_val:6.1f},{p.y_val:6.1f}) alt={-p.z_val:5.1f} m  speed={spd:4.1f} m/s",
                      end="", flush=True)
                # moveOnPathAsync finishes when the drone is back near the last waypoint
                if time.time() - t0 > 3 and math.hypot(p.x_val - path[-1].x_val, p.y_val - path[-1].y_val) < 2.5:
                    break
                time.sleep(0.5)
            f.join()
            print()

        print("Returning home and landing...")
        c.moveToPositionAsync(0, 0, -abs(args.alt), 5, vehicle_name=v).join()
        c.landAsync(vehicle_name=v).join()
    except KeyboardInterrupt:
        print("\nInterrupted - landing.")
        c.landAsync(vehicle_name=v).join()
    finally:
        c.armDisarm(False, v)
        c.enableApiControl(False, v)
        print("Done.")


if __name__ == "__main__":
    main()
