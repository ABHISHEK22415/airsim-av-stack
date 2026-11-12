#!/usr/bin/env python3
"""Test scenario for the AV stack: people on sidewalks, pedestrians that CROSS the street on the tour
route, and static obstacles (broken-down vehicles / cones) blocking the driving lane.

    .venv/bin/python scenario.py                 # until Ctrl+C (removes everything on exit)
    .venv/bin/python scenario.py --walkers 30 --no-obstacles
"""
import argparse
import math
import random
import signal
import time

import airsim

from drive_to import EDGES, add, right, unit
from people import Walker

# lane-blocking obstacles on the tour route (asset, x, y, yaw) - in the right-hand lane of travel direction
OBSTACLES = [
    ("boxtruck_WithHeadlights_boxtruck_fixedtransforms_LOD0", 45.0, 0.6, 0.0),   # (0,-1)->(80,-1), heading +x
    ("Car_01", 126.4, 60.0, 0.0),    # yaw 0 = length along y for this mesh (1.57 put it ACROSS the road)
    ("Cone", -60.0, 125.4, 0.0), ("Cone", -62.0, 125.9, 0.0), ("Cone", -64.0, 125.4, 0.0),  # (0,127)->(-128,127)
    ("Vehicle_Police_ISM", -126.4, -70.0, -1.57),                               # (-128,-1)->(-128,-129)
]
# street crossings on the tour route: (street a, street b, distance from a)
CROSSINGS = [((80, -1), (128, -1), 25.0), ((0, 127), (-128, 127), 30.0), ((-128, -129), (0, -129), 64.0)]


class Crosser:
    def __init__(self, a, b, s):
        d = unit(a, b)
        self.mid = add(a, d, s)
        self.r = right(d)
        self.side = random.choice([-1, 1])
        self.u = self.side * 7.5          # lateral position (m) across the street
        self.wait = random.uniform(2, 8)
        self.speed = 1.4

    def step(self, dt):
        if self.wait > 0:
            self.wait -= dt
        else:
            target = -self.side * 7.5
            self.u += math.copysign(self.speed * dt, target - self.u)
            if abs(self.u - target) < 0.1:
                self.side, self.wait = -self.side, random.uniform(8, 16)
        p = add(self.mid, self.r, self.u)
        heading = (-self.r[0] * self.side, -self.r[1] * self.side)
        return airsim.Pose(airsim.Vector3r(p[0], p[1], 0), airsim.to_quaternion(0, 0, math.atan2(heading[1], heading[0])))


def _term(*_):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, _term)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walkers", type=int, default=30)
    ap.add_argument("--no-obstacles", action="store_true")
    ap.add_argument("--no-crossers", action="store_true")
    args = ap.parse_args()
    c = airsim.CarClient()
    c.confirmConnection()
    random.seed(5)
    movers, spawned = [], []
    edges = sorted(EDGES)
    for i in range(args.walkers):
        w = Walker(edges[i % len(edges)], random.choice([-1, 1]), random.uniform(1.5, 4.0))
        w.name = c.simSpawnObject(f"walker_{i}", "BicycleMan", w.step(0), airsim.Vector3r(1, 1, 1), False, False)
        movers.append(w)
        spawned.append(w.name)
    if not args.no_crossers:
        for i, (a, b, s) in enumerate(CROSSINGS):
            x = Crosser(a, b, s)
            x.name = c.simSpawnObject(f"crosser_{i}", "BicycleMan", x.step(0), airsim.Vector3r(1, 1, 1), False, False)
            movers.append(x)
            spawned.append(x.name)
    if not args.no_obstacles:
        for i, (asset, x, y, yaw) in enumerate(OBSTACLES):
            scale = airsim.Vector3r(0.6, 0.6, 0.9) if asset == "Cone" else airsim.Vector3r(1, 1, 1)
            z = -0.45 if asset == "Cone" else 0.0
            spawned.append(c.simSpawnObject(f"obstacle_{i}", asset, airsim.Pose(airsim.Vector3r(x, y, z),
                                            airsim.to_quaternion(0, 0, yaw)), scale, False, False))
    print(f"Scenario: {args.walkers} walkers, {0 if args.no_crossers else len(CROSSINGS)} street crossers, "
          f"{0 if args.no_obstacles else len(OBSTACLES)} lane obstacles.", flush=True)
    last = time.time()
    try:
        while True:
            now = time.time()
            dt, last = now - last, now
            for m in movers:
                c.simSetObjectPose(m.name, m.step(dt), True)
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        for n in spawned:
            c.simDestroyObject(n)
        print("Scenario removed.")


if __name__ == "__main__":
    main()
