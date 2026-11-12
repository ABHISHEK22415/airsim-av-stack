#!/usr/bin/env python3
"""Fill AirSimNH streets with people (the map's only human model is "BicycleMan", a man on a bike)
moving back and forth along the sidewalks of every street.
(Deer blueprint "DeerBothBP" crashes the sim when spawned via API - do not use.)

    .venv/bin/python people.py               # 40 people, runs until Ctrl+C (then removes them)
    .venv/bin/python people.py --n 60 --secs 600

People stay on the sidewalks (6.8 m from the street centre line; the car drives 1.9 m right of centre)
and turn around before junctions, so they never cross the car's path. Kinematic: moved with simSetObjectPose.
"""
import argparse
import math
import random
import time

import airsim

from drive_to import EDGES, JUNCTION_R, unit, right, add

SIDEWALK = 6.8


class Walker:
    def __init__(self, edge, side, speed):
        a, b = edge
        self.a, self.d = a, unit(a, b)
        self.r = right(self.d)
        self.side, self.speed = side, speed
        self.lo, self.hi = JUNCTION_R + 3, math.dist(a, b) - JUNCTION_R - 3
        self.s = random.uniform(self.lo, self.hi)
        self.dir = random.choice([-1, 1])
        self.name = None

    def step(self, dt):
        self.s += self.dir * self.speed * dt
        if self.s > self.hi:
            self.s, self.dir = self.hi, -1
        elif self.s < self.lo:
            self.s, self.dir = self.lo, 1
        p = add(add(self.a, self.d, self.s), self.r, self.side * SIDEWALK)
        yaw = math.atan2(self.d[1] * self.dir, self.d[0] * self.dir)
        return airsim.Pose(airsim.Vector3r(p[0], p[1], 0), airsim.to_quaternion(0, 0, yaw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of people")
    ap.add_argument("--secs", type=float, default=0, help="0 = until Ctrl+C")
    args = ap.parse_args()

    c = airsim.CarClient()
    c.confirmConnection()
    random.seed(3)
    edges = sorted(EDGES)
    walkers, spawned = [], []
    for i in range(args.n):
        w = Walker(edges[i % len(edges)], random.choice([-1, 1]), random.uniform(1.5, 4.5))
        w.name = c.simSpawnObject(f"person_{i}", "BicycleMan", w.step(0), airsim.Vector3r(1, 1, 1), False, False)
        walkers.append(w)
        spawned.append(w.name)
    print(f"Spawned {args.n} people. Ctrl+C to stop.", flush=True)

    t_end = time.time() + args.secs if args.secs else float("inf")
    last = time.time()
    try:
        while time.time() < t_end:
            now = time.time()
            dt, last = now - last, now
            for w in walkers:
                c.simSetObjectPose(w.name, w.step(dt), True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        for n in spawned:
            c.simDestroyObject(n)
        print("Removed people.")


if __name__ == "__main__":
    main()
