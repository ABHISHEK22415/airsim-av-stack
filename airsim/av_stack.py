#!/usr/bin/env python3
"""AV stack v2 for AirSimNH  (baseline v1 = drive_to.py)

  PERCEPTION  6-camera Tesla-style rig, RCCC sensor model, YOLOv8 objects, map-guided RCCC traffic-light
              classifier, camera->ground "vector space" projection            (perception.py, own process)
              16-channel LiDAR: ground removal + obstacle points in world frame
  PLANNING    route: Dijkstra on the street graph (city loop)
              local: LATTICE planner - lateral-offset candidates (quintic transitions) around the lane centre,
              collision-checked against LiDAR points, cost = offset + change + opposite-lane + collision
              behaviour: traffic lights from camera (map/V2X fallback), pedestrian yield (YOLO + LiDAR),
              obstacle stop when no candidate is free
              speed: curvature-limited profile + stop-line / obstacle / goal braking curves
  CONTROL     STANLEY lateral controller (front-axle cross-track + heading error)
              PI longitudinal controller with anti-windup
  DISPLAY     in-sim: city route map, lattice candidates (grey) + chosen path (green), obstacle points (red)
              dashboard window: 6 RCCC camera tiles + live route map (BEV) + telemetry

    .venv/bin/python av_stack.py --laps 1
"""
import argparse
import math
import multiprocessing as mp
import os
import time

import airsim
import cv2
import numpy as np

import drive_to as base
base.LANE = 1.6   # a bit closer to the centre line than v1 (v1 clipped parked cars at 1.9 m)
from drive_to import ADJ, EDGES, NODES, SIGNALS, add, build_path, dijkstra, light, right, unit  # noqa: E402
from perception import perception_proc  # noqa: E402
from signals import lamp_pos  # noqa: E402

LOOP = [(0, -1), (80, -1), (128, -1), (128, 127), (0, 127), (-128, 127), (-128, -1), (-128, -129), (0, -129), (0, -1)]
CLEAR = 1.3                       # car half width 0.95 + 0.35 margin
OFFSETS = [round(o, 1) for o in np.arange(0.4, -4.45, -0.4)]   # 0.4 m lattice resolution
CLEAR_TIGHT = 1.1                 # squeeze margin (slow creep only)
HORIZON = 45.0
A_BRAKE, A_LAT = 3.5, 2.4
WHEELBASE = 2.8


# ------------------------------------------------------------------ helpers
def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def smooth5(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (10 - 15 * x + 6 * x * x)


def lidar_obstacles(c, car_xy, yaw=None):
    d = c.getLidarData("Lidar1")
    if len(d.point_cloud) < 30:
        return np.zeros((0, 2))
    p = np.array(d.point_cloud, dtype=np.float32).reshape(-1, 3)
    ground = np.percentile(p[:, 2], 92)          # NED: ground = largest z
    h = ground - p[:, 2]
    m = (h > 0.3) & (h < 2.4)
    xy = p[m, :2].astype(np.float64)
    if yaw is None:
        xy = xy[np.hypot(xy[:, 0] - car_xy[0], xy[:, 1] - car_xy[1]) > 2.6]   # drop own body (old: circle)
    else:   # drop own body only: box 2.6 m long x 1.15 m half-width (keeps obstacles right beside the car)
        rel = xy - car_xy
        lon = rel[:, 0] * math.cos(yaw) + rel[:, 1] * math.sin(yaw)
        lat = -rel[:, 0] * math.sin(yaw) + rel[:, 1] * math.cos(yaw)
        xy = xy[(np.abs(lon) > 2.6) | (np.abs(lat) > 1.15)]
    if len(xy):                                  # 0.3 m voxel downsample
        _, idx = np.unique(np.round(xy / 0.3).astype(np.int32), axis=0, return_index=True)
        xy = xy[idx]
    return xy


class Tracker:
    """LiDAR multi-object tracker: grid clustering -> nearest-neighbour association (with constant-velocity
    prediction) -> exponentially smoothed velocity.  Returns confirmed MOVING tracks (pedestrians, cyclists, animals)."""

    def __init__(self):
        self.tracks, self.next_id = [], 0

    @staticmethod
    def cluster(xy, cell=0.8):
        if len(xy) == 0:
            return []
        cells = {}
        for i, key in enumerate(map(tuple, np.floor(xy / cell).astype(int))):
            cells.setdefault(key, []).append(i)
        seen, out = set(), []
        for key in cells:
            if key in seen:
                continue
            stack, idx = [key], []
            seen.add(key)
            while stack:
                a = stack.pop()
                idx += cells[a]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        b = (a[0] + dx, a[1] + dy)
                        if b in cells and b not in seen:
                            seen.add(b)
                            stack.append(b)
            pts = xy[idx]
            out.append((pts.mean(0), float(np.ptp(pts, 0).max()), len(idx)))
        return out

    def update(self, xy, t, car_xy):
        if len(xy):
            xy = xy[np.hypot(xy[:, 0] - car_xy[0], xy[:, 1] - car_xy[1]) < 35]   # dense LiDAR range only
        dets = [c_ for c_ in self.cluster(xy) if c_[1] < 2.5 and c_[2] >= 3]   # person/bike/animal sized
        new, used = [], set()
        for cen, ext, n in dets:
            best, bd = None, 1.5
            for j, tr in enumerate(self.tracks):
                if j in used:
                    continue
                d = np.linalg.norm(cen - (tr["c"] + tr["v"] * (t - tr["t"])))
                if d < bd:
                    bd, best = d, j
            if best is None:
                new.append(dict(id=self.next_id, c=cen, v=np.zeros(2), t=t, hits=1, hist=[(t, cen)], fit=9.0))
                self.next_id += 1
                continue
            tr = self.tracks[best]
            used.add(best)
            tr["hist"] = [h for h in tr["hist"] if t - h[0] < 1.2] + [(t, cen)]
            tr["c"], tr["t"], tr["hits"] = cen, t, tr["hits"] + 1
            T = np.array([h[0] for h in tr["hist"]])
            if len(T) >= 5 and T[-1] - T[0] > 0.6:       # least-squares constant-velocity fit
                C = np.array([h[1] for h in tr["hist"]])
                A = np.stack([T - T.mean(), np.ones_like(T)], 1)
                sol, *_ = np.linalg.lstsq(A, C, rcond=None)
                tr["v"] = sol[0]
                tr["fit"] = float(np.sqrt(np.mean(np.sum((A @ sol - C) ** 2, 1))))
            new.append(tr)
        self.tracks = new
        return [tr for tr in self.tracks if tr["hits"] >= 5 and tr["fit"] < 0.25
                and 0.5 < np.linalg.norm(tr["v"]) < 4.5]


# ------------------------------------------------------------------ route map (BEV) image
class MapView:
    def __init__(self, path, size=480, lo=-145, hi=145):
        self.size, self.lo, self.hi = size, lo, hi
        self.k = size / (hi - lo)
        bg = np.full((size, size, 3), (34, 38, 30), np.uint8)
        for a, b in EDGES:
            cv2.line(bg, self.px(a), self.px(b), (90, 90, 90), max(2, int(12 * self.k)))
        for a, b in EDGES:
            cv2.line(bg, self.px(a), self.px(b), (0, 180, 220), 1)
        cv2.polylines(bg, [np.array([self.px(p) for p in path[::3]], np.int32)], False, (255, 200, 0), 2)
        for n in NODES:
            cv2.circle(bg, self.px(n), 3, (200, 200, 200), -1)
        self.bg = bg
        self.path = path

    def px(self, p):  # NED x (north) up, y (east) right
        return int((p[1] - self.lo) * self.k), int((self.hi - p[0]) * self.k)

    def render(self, car, yaw, idx, obs, objs, now, cand=None):
        im = self.bg.copy()
        cv2.polylines(im, [np.array([self.px(p) for p in self.path[:idx + 1:3]], np.int32)], False, (120, 120, 60), 2)
        for n in SIGNALS:
            for m in ADJ[n]:
                d = unit(m, n)
                st = light(n, "x" if abs(d[0]) > 0.5 else "y", now)
                q = add(add(n, d, -11), right(d), 4)
                cv2.circle(im, self.px(q), 3, {"R": (0, 0, 255), "Y": (0, 220, 255), "G": (0, 255, 0)}[st], -1)
        for x, y in obs[::2]:
            cv2.circle(im, self.px((x, y)), 1, (60, 60, 255), -1)
        for name, x, y, cam in objs:
            col = (0, 200, 255) if name in ("person", "bicycle") else (255, 160, 0)
            cv2.circle(im, self.px((x, y)), 4, col, 1)
        if cand is not None:
            cv2.polylines(im, [np.array([self.px(p) for p in cand], np.int32)], False, (0, 255, 0), 2)
        h = (math.cos(yaw), math.sin(yaw))
        r = right(h)
        tri = [add(car, h, 5), add(add(car, h, -3), r, 2.5), add(add(car, h, -3), r, -2.5)]
        cv2.fillPoly(im, [np.array([self.px(p) for p in tri], np.int32)], (255, 255, 255))
        cv2.putText(im, "ROUTE MAP  (N up)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, "cyan=route  red=LiDAR  orange/blue=camera objects", (8, self.size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        return im


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--speed", type=float, default=8.5)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--planner", default="v3", choices=["v2", "v3"], help="lattice cost function")
    ap.add_argument("--tag", default="run", help="label for logs/runs/*.json")
    ap.add_argument("--stream-file", default="", help="also write each dashboard frame as JPEG here")
    args = ap.parse_args()

    c = airsim.CarClient()
    c.confirmConnection()
    c.enableApiControl(True)
    st = c.getCarState().kinematics_estimated
    pos = (st.position.x_val, st.position.y_val)
    yaw = airsim.to_eularian_angles(st.orientation)[2]
    start = min(NODES, key=lambda n: math.dist(n, pos))
    lead = dijkstra(start, LOOP[0], (math.cos(yaw), math.sin(yaw))) if start != LOOP[0] else [LOOP[0]]
    route = lead[:-1] + LOOP[:-1] * args.laps + [(0, -1), (80, -1)]
    path, S, stops = build_path(pos, route, (40, -1))
    P, S = np.array(path), np.array(S)
    hd = np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0]))
    NR = np.stack([-np.sin(hd), np.cos(hd)], 1)             # right normals
    dh = np.abs(np.array([wrap(a) for a in np.diff(hd, append=hd[-1])]))
    kappa = np.convolve(dh / np.maximum(np.diff(S, append=S[-1] + 1), 0.2), np.ones(5) / 5, "same")
    v_curve = np.sqrt(A_LAT / np.maximum(kappa, 1e-3))
    for s in stops:   # approach direction for the lamp lookup
        j = int(np.searchsorted(S, s[0]))
        s.append((round(math.cos(hd[j])), round(math.sin(hd[j]))))
    print(f"Route: {len(route)} junctions, {S[-1]:.0f} m, {len(stops)} signalised stop lines", flush=True)

    # in-sim: whole street network + route
    c.simFlushPersistentMarkers()
    net = []
    for a, b in EDGES:
        net += [airsim.Vector3r(a[0], a[1], -0.1), airsim.Vector3r(b[0], b[1], -0.1)]
    c.simPlotLineList(net, color_rgba=[1, 0.85, 0, 1], thickness=4, is_persistent=True)
    c.simPlotLineStrip([airsim.Vector3r(x, y, -0.2) for x, y in path[::2]], color_rgba=[0.1, 0.8, 1, 0.8],
                       thickness=6, is_persistent=True)

    # perception process
    lamp_arr = mp.Array("d", [0, 0, 0, 0])
    q, stop_evt = mp.Queue(maxsize=2), mp.Event()
    proc = mp.Process(target=perception_proc, args=(lamp_arr, q, stop_evt), daemon=True)
    proc.start()
    percep = dict(t=0, tl=(None, 0, 0, None), objs=[], counts={}, fps=0, front=[], grid=None)
    grid_img = np.zeros((675, 800, 3), np.uint8)

    mapv = MapView(path)
    if not args.no_dashboard:
        cv2.namedWindow("AV stack v2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AV stack v2", 1280, 675)
        cv2.moveWindow("AV stack v2", 2000, 60)

    ctl = airsim.CarControls()
    tracker, movers = Tracker(), []
    idx, prev_off, I_err, steer_prev = 0, 0.0, 0.0, 0.0
    man = dict(off=0.0, s0=0.0, d0=0.0, Lt=10.0)
    tl_hist, tl_last_t, snap_t = [], 0.0, 0.0
    blocked_since = None
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)
    obs = np.zeros((0, 2))
    t0 = last = time.time()
    k = 0
    stats = dict(recoveries=0, collisions=set(), red_violations=0, cam_tl=0, cam_tl_ok=0, fallback=0, xte=[], swerves=0, ped_stops=0,
                 wrong_m=0.0, linger_m=0.0, excursions=[])   # oncoming-lane metrics
    exc, s_prev = None, None
    mem, obs_plan = [], np.zeros((0, 2))   # current oncoming-lane excursion: dict(s0, t0, linger)
    last_col, crossed = None, set()
    try:
        while True:
            now = time.time()
            dt, last = max(1e-3, now - last), now
            k += 1
            cs = c.getCarState()
            kin = cs.kinematics_estimated
            x, y = kin.position.x_val, kin.position.y_val
            yaw = airsim.to_eularian_angles(kin.orientation)[2]
            v = cs.speed
            if k % 2 == 0:
                if args.planner == "v2":
                    obs = lidar_obstacles(c, (x, y))
                else:
                    obs = lidar_obstacles(c, (x, y), yaw)
                    # 2. memory: static points that left the front-180 deg LiDAR FOV (beside/behind the car) are
                    #    kept for 4 s so the car does not cut back in while still alongside an obstacle
                    hv = np.array([math.cos(yaw), math.sin(yaw)])
                    mem = [(t_, p_) for t_, p_ in mem if now - t_ < 4.0]
                    if len(obs):
                        mem.append((now, obs[(obs - (x, y)) @ hv < 3.0]))
                    old = [p_[((p_ - (x, y)) @ hv < 0.5) & (np.hypot(*(p_ - (x, y)).T) < 12)] for t_, p_ in mem[:-1]]
                    old = [o_ for o_ in old if len(o_)]
                    obs_plan = np.vstack([obs] + old) if old else obs
                movers = tracker.update(obs, now, (x, y))
            while not q.empty():
                percep = q.get_nowait()
                if percep.get("grid"):
                    grid_img = cv2.imdecode(np.frombuffer(percep["grid"], np.uint8), cv2.IMREAD_COLOR)

            # ---- localisation on reference path
            lo, hi = idx, min(idx + 60, len(P))
            idx = lo + int(np.argmin(((P[lo:hi] - (x, y)) ** 2).sum(1)))
            s_cur = S[idx]
            d0 = float(np.dot((x, y) - P[idx], NR[idx]))
            stats["xte"].append(abs(d0 - prev_off))

            # ---- lattice candidates
            ss = np.arange(0.0, HORIZON, 1.0)
            ii = np.clip(np.searchsorted(S, s_cur + ss), 0, len(P) - 1)
            Lt = max(10.0, 1.6 * v + 6)
            ob_ = obs if args.planner == "v2" else obs_plan
            near = ob_[np.hypot(ob_[:, 0] - x, ob_[:, 1] - y) < HORIZON + 8] if len(ob_) else ob_
            # motion prediction: moving tracks are swept 3 s ahead (constant velocity) and added as obstacles
            pred_pts = [tr["c"] + tr["v"] * tp for tr in movers for tp in np.arange(0.3, 3.01, 0.3)]
            if pred_pts:
                near = np.vstack([near.reshape(-1, 2), np.array(pred_pts)])
            # vulnerable road users near/entering the lane ahead (LiDAR tracks + camera people)
            ahead_pts = P[idx:int(np.searchsorted(S, s_cur + 45))]
            vru = False
            if len(ahead_pts):
                cand_vru = [tr["c"] + tr["v"] * tp for tr in movers for tp in (0.0, 1.0, 2.0, 3.0)]
                cand_vru += [np.array([ox, oy]) for nm, ox, oy, _ in percep["objs"]
                             if nm in ("person", "bicycle") and now - percep["t"] < 1.0]
                for q_ in cand_vru:
                    if np.min(np.hypot(ahead_pts[:, 0] - q_[0], ahead_pts[:, 1] - q_[1])) < 3.5:
                        vru = True
                        break
            # pedestrian yield: camera sees a person/bicycle in the front corridor -> no swerving, stop for it
            corridor = {"cam_front_narrow": (220, 420, 45), "cam_front_wide": (270, 370, 30)}  # u range, min box height
            ped = any(n in ("person", "bicycle") and corridor[cam][0] < (x1 + x2) / 2 < corridor[cam][1]
                      and (y2 - y1) > corridor[cam][2] for n, x1, y1, x2, y2, cam in percep["front"]) \
                and now - percep["t"] < 0.8
            offsets = [0.0] if ped else OFFSETS
            # anchored manoeuvre: lateral profile fixed in path coordinates (s), so it does not "slide"
            # ahead of the car every cycle.  New candidates start from the currently PLANNED lateral offset.
            d_plan = float(man["d0"] + (man["off"] - man["d0"]) * smooth5((s_cur - man["s0"]) / man["Lt"]))
            def evaluate(clear, Lt_):
                out = []
                for off in offsets:
                    if off == man["off"]:
                        lat = man["d0"] + (off - man["d0"]) * smooth5((s_cur + ss - man["s0"]) / man["Lt"])
                    else:
                        lat = d_plan + (off - d_plan) * smooth5(ss / Lt_)
                    pts = P[ii] + NR[ii] * lat[:, None]
                    s_col = None
                    if len(near):
                        if args.planner == "v2":
                            chk = pts
                        else:   # body positions: lateral at s+-k m applied at s (rear/front of the car)
                            jj = np.arange(len(lat))
                            lat_b = np.concatenate([np.full(3, d_plan), lat])   # 3 m behind the car = current offset
                            chk = np.stack([P[ii] + NR[ii] * lat_b[np.clip(jj + 3 + k_, 0, len(lat_b) - 1)][:, None]
                                            for k_ in (-2, 0, 2)], 1)          # (H, 3, 2)
                            chk = chk.reshape(-1, 2)
                        dmin = np.sqrt(((chk[:, None, :] - near[None, :, :]) ** 2).sum(-1)).min(1)
                        if args.planner != "v2":
                            dmin = dmin.reshape(len(lat), -1).min(1)
                        hit = np.nonzero(dmin < clear)[0]
                        if len(hit):
                            s_col = float(ss[hit[0]])
                    if args.planner == "v2":   # old: change penalty > lane penalty -> stays in the oncoming lane
                        cost = 0.35 * abs(off) + 1.2 * abs(off - man["off"]) + (3.0 if off <= -3 else 0.0) + (2.0 if off <= -4 else 0.0)
                    else:   # v3: own lane dominates, so the car returns as soon as its lane is clear
                        cost = 1.5 * abs(off) + 0.4 * abs(off - man["off"]) + (2.5 if off < -1.0 else 0.0) \
                            + (2.0 if off <= -3 else 0.0)
                    if s_col is not None:
                        cost += 100 + 6 * (HORIZON - s_col)
                    out.append((cost, off, pts, s_col))
                return out

            cands = evaluate(CLEAR, Lt)
            squeeze = False
            if all(c_[3] is not None and c_[3] < 15 for c_ in cands) and v < 3.0 and not ped:
                tight = evaluate(CLEAR_TIGHT, 6.0)          # tight-squeeze mode: smaller margin, short swerve
                if any(c_[3] is None for c_ in tight):
                    cands, squeeze, Lt = tight, True, 6.0
            best = min(cands, key=lambda c_: c_[0])
            keep = next((c_ for c_ in cands if c_[1] == man["off"]), None)
            if keep and keep[0] <= best[0] + 0.8:
                best = keep
            if best[1] != man["off"]:
                if best[1] < -1.5 and man["off"] > -1.5:
                    stats["swerves"] += 1
                man = dict(off=best[1], s0=s_cur, d0=d_plan, Lt=Lt)
            _, off, cpts, s_col = best
            prev_off = off
            stats["xte"][-1] = abs(d0 - d_plan)

            # ---- oncoming-lane metrics: body over the centre line (lane centre -1.9 m, half width 0.95 m)
            ds_ = 0.0 if s_prev is None else max(0.0, s_cur - s_prev)
            s_prev = s_cur
            home_free = any(c_[1] == 0.0 and c_[3] is None for c_ in cands)   # own lane clear over the horizon
            if d0 < -0.95:
                if exc is None:
                    exc = dict(s0=s_cur, t0=now, linger=0.0)
                stats["wrong_m"] += ds_
                if home_free:
                    exc["linger"] += ds_
                    stats["linger_m"] += ds_
            elif exc is not None:
                stats["excursions"].append(dict(len_m=round(s_cur - exc["s0"], 1), dur_s=round(now - exc["t0"], 1),
                                                linger_m=round(exc["linger"], 1), at_s=round(exc["s0"])))
                exc = None

            # ---- behaviour: speed limits
            vt = args.speed
            ahead = (S >= s_cur) & (S < s_cur + 45)
            if ahead.any():
                vt = min(vt, float(np.min(np.sqrt(v_curve[ahead] ** 2 + 2 * A_BRAKE * (S[ahead] - s_cur)))))
            reason = "cruise"
            if vru and vt > 7.0:
                vt, reason = 7.0, "VRU CAUTION 25 km/h"
            if s_col is not None:   # no free candidate -> stop before the obstacle
                v_obs = math.sqrt(max(0.0, 2 * A_BRAKE * (s_col - 6.0)))
                if v_obs < vt:
                    vt, reason = v_obs, ("PEDESTRIAN YIELD" if ped else "OBSTACLE STOP") + f" {s_col:.0f} m"
                    if ped and v < 0.5:
                        stats["ped_stops"] += 1 if k % 40 == 0 else 0
            elif off != 0.0:
                reason = f"LATTICE AVOID offset {off:+.1f} m"
            if squeeze:
                vt, reason = min(vt, 2.5), f"TIGHT SQUEEZE offset {off:+.1f} m"
            # deadlock recovery: blocked & stopped for > 6 s -> back up ~3 m to make room for a swerve
            if s_col is not None and v < 0.3 and not ped:
                blocked_since = blocked_since or now
                if now - blocked_since > 6.0:
                    print("\n[deadlock - reversing to re-plan]", flush=True)
                    stats["recoveries"] += 1
                    c.setCarControls(airsim.CarControls(throttle=0.5, is_manual_gear=True, manual_gear=-1))
                    time.sleep(3.2)
                    c.setCarControls(airsim.CarControls(brake=1))
                    time.sleep(0.6)
                    blocked_since, I_err = None, 0.0
                    idx = max(0, idx - 8)
                    last = time.time()
                    continue
            else:
                blocked_since = None
            # traffic light
            tl_txt = "no signal"
            lamp_arr[3] = 0
            for sl in stops:
                dist = sl[0] - s_cur
                if dist < -1:
                    if id(sl) not in crossed:
                        crossed.add(id(sl))
                        if light(sl[1], sl[2], now) == "R" and sl[3] != "go":
                            stats["red_violations"] += 1
                    continue
                if dist > 75:
                    break
                lx, ly, lz = lamp_pos(sl[1], sl[4])
                lamp_arr[0], lamp_arr[1], lamp_arr[2], lamp_arr[3] = lx, ly, lz, 1
                truth = light(sl[1], sl[2], now)
                cam_now = percep["tl"][0] if (now - percep["t"] < 0.8 and percep["tl"][3]) else None
                if cam_now and percep["t"] != tl_last_t:          # temporal vote over last ~0.8 s of frames
                    tl_hist.append((percep["t"], cam_now, sl[0]))
                    tl_last_t = percep["t"]
                votes = [st_ for t_, st_, s_ in tl_hist if now - t_ < 0.8 and s_ == sl[0]]
                cam_state = max(set(votes), key=votes.count) if votes else None
                if cam_state:
                    state, src = cam_state, f"CAMERA RCCC R/C={percep['tl'][1]:.2f}"
                    stats["cam_tl"] += 1
                    stats["cam_tl_ok"] += cam_state == truth
                else:
                    state, src = truth, "map/V2X fallback"
                    stats["fallback"] += 1
                name = {"G": "GREEN", "Y": "YELLOW", "R": "RED"}[state]
                tl_txt = f"{name} in {dist:.0f} m [{src}]"
                must = state == "R" or (state == "Y" and sl[3] != "go")
                if state == "Y" and sl[3] is None:
                    sl[3] = "go" if dist < v * v / (2 * 3.0) + 1 else "stop"
                    must = sl[3] == "stop"
                if state == "G":
                    sl[3] = None
                if must:
                    v_tl = math.sqrt(max(0.0, 2 * A_BRAKE * (dist - 1.0)))
                    if v_tl < vt:
                        vt, reason = v_tl, f"{name} LIGHT"
                break
            d_goal = S[-1] - s_cur
            vt = min(vt, math.sqrt(max(0.0, 2 * 2.0 * (d_goal - 0.5))))

            # ---- STANLEY lateral control on the chosen lattice path
            fx, fy = x + math.cos(yaw) * WHEELBASE / 2, y + math.sin(yaw) * WHEELBASE / 2
            j = int(np.argmin(((cpts - (fx, fy)) ** 2).sum(1)))
            j = min(j, len(cpts) - 2)
            th = math.atan2(cpts[j + 1][1] - cpts[j][1], cpts[j + 1][0] - cpts[j][0])
            e = (fx - cpts[j][0]) * -math.sin(th) + (fy - cpts[j][1]) * math.cos(th)   # + = right of path
            psi = wrap(th - yaw)
            delta = psi + math.atan2(-1.6 * e, v + 1.5)
            steer = max(-1.0, min(1.0, delta / 0.6))
            steer = 0.6 * steer + 0.4 * steer_prev
            steer_prev = steer
            ctl.steering = steer

            # ---- PI speed control
            err = vt - v
            if vt < 0.25 and v < 1.2:
                ctl.throttle, ctl.brake, I_err = 0.0, 1.0, 0.0
            else:
                I_err = max(-4, min(4, I_err + err * dt))
                u = 0.32 * err + 0.06 * I_err + 0.025 * vt
                if u >= 0:
                    ctl.throttle, ctl.brake = min(1.0, u), 0.0
                else:
                    ctl.throttle, ctl.brake = 0.0, min(1.0, -0.6 * u)
            c.setCarControls(ctl)

            ci = c.simGetCollisionInfo()
            if ci.has_collided and ci.time_stamp != last_col and now - t0 > 2:
                last_col = ci.time_stamp
                if ci.object_name not in stats["collisions"]:
                    print(f"\n[collision with {ci.object_name}]", flush=True)
                stats["collisions"].add(ci.object_name)

            # ---- visualisation
            if k % 4 == 0:
                for _, o, pts, sc in cands:
                    if o != off:
                        c.simPlotLineStrip([airsim.Vector3r(px_, py_, -0.35) for px_, py_ in pts[::2]],
                                           color_rgba=[0.6, 0.6, 0.6, 0.6] if sc is None else [1, 0.3, 0.3, 0.6],
                                           thickness=3, duration=0.25)
                c.simPlotLineStrip([airsim.Vector3r(px_, py_, -0.4) for px_, py_ in cpts], color_rgba=[0.1, 1, 0.2, 1],
                                   thickness=10, duration=0.25)
                c.simPrintLogMessage("AV  ", f"{v*3.6:4.1f}/{vt*3.6:4.1f} km/h | {reason} | goal {d_goal:4.0f} m", 0)
                c.simPrintLogMessage("TL  ", tl_txt, 0)
                c.simPrintLogMessage("CAM ", f"YOLO {percep['counts']} @ {percep['fps']:.1f} Hz | LiDAR obstacles {len(near)}", 0)
            if not args.no_dashboard and k % 2 == 0:
                mp_img = mapv.render((x, y), yaw, idx, near, percep["objs"], now, cpts)
                for tr in movers:
                    cv2.arrowedLine(mp_img, mapv.px(tuple(tr['c'])), mapv.px(tuple(tr['c'] + tr['v'] * 3)), (0, 255, 255), 2)
                g = cv2.resize(grid_img, (800, 675)) if grid_img.shape[:2] != (675, 800) else grid_img
                right_panel = np.zeros((675, 480, 3), np.uint8)
                right_panel[:480] = mp_img
                lines = [f"speed {v*3.6:5.1f} km/h  target {vt*3.6:5.1f}", f"mode: {reason}", f"light: {tl_txt}",
                         f"lattice offset {off:+.1f} m  xte {e:+.2f} m  psi {math.degrees(psi):+.1f} deg",
                         f"Stanley steer {steer:+.2f}   PI throttle {ctl.throttle:.2f} brake {ctl.brake:.2f}",
                         f"goal in {d_goal:5.0f} m   collisions {len(stats['collisions'])}",
                         f"camera {percep['fps']:.1f} Hz  objects {sum(percep['counts'].values())}"]
                for i_, t_ in enumerate(lines):
                    cv2.putText(right_panel, t_, (8, 500 + 24 * i_), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (230, 230, 230), 1,
                                cv2.LINE_AA)
                dash = np.hstack([g, right_panel])
                cv2.imshow("AV stack v2", dash)
                if args.stream_file:   # for stream_server.py (atomic replace so readers never see half a JPEG)
                    ok_, buf_ = cv2.imencode(".jpg", dash, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok_:
                        with open(args.stream_file + ".tmp", "wb") as f_:
                            f_.write(buf_.tobytes())
                        os.replace(args.stream_file + ".tmp", args.stream_file)
                if now - snap_t > 5:
                    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "dashboard_latest.jpg"), dash)
                    snap_t = now
                cv2.waitKey(1)
            if k % 10 == 0:
                print(f"\r t={now-t0:5.0f}s {v*3.6:5.1f} km/h | {reason:28s} | {tl_txt:40s} | goal {d_goal:5.0f} m",
                      end="", flush=True)
            if d_goal < 1.5 and v < 0.3:
                print("\nArrived.")
                break
            time.sleep(max(0.0, 0.05 - (time.time() - now)))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        c.setCarControls(airsim.CarControls(brake=1))
        c.enableApiControl(False)
        stop_evt.set()
        proc.join(timeout=3)
        cv2.destroyAllWindows()
        xte = np.array(stats["xte"]) if stats["xte"] else np.zeros(1)
        print("\n==== run summary ====")
        print(f"time {time.time()-t0:.0f} s, distance {S[min(idx, len(S)-1)]:.0f} m")
        print(f"collisions: {len(stats['collisions'])} {sorted(stats['collisions'])}")
        print(f"red-light violations: {stats['red_violations']}")
        tot = stats["cam_tl"] + stats["fallback"]
        if tot:
            print(f"traffic light source: camera {100*stats['cam_tl']/tot:.0f}% (agree with truth "
                  f"{100*stats['cam_tl_ok']/max(1,stats['cam_tl']):.0f}%), fallback {100*stats['fallback']/tot:.0f}%")
        print(f"lattice swerves into left lane: {stats['swerves']}, deadlock recoveries: {stats['recoveries']}")
        print(f"lateral tracking error vs chosen offset: mean {xte.mean():.2f} m, 95% {np.percentile(xte,95):.2f} m")
        if exc is not None:
            stats["excursions"].append(dict(len_m=round(S[min(idx, len(S)-1)] - exc["s0"], 1), dur_s=round(time.time() - exc["t0"], 1),
                                            linger_m=round(exc["linger"], 1), at_s=round(exc["s0"]), unfinished=True))
        dist = float(S[min(idx, len(S)-1)])
        exs = stats["excursions"]
        print(f"oncoming lane: {stats['wrong_m']:.0f} m ({100*stats['wrong_m']/max(dist,1):.1f}% of distance), "
              f"{len(exs)} excursions, max {max([e['len_m'] for e in exs], default=0):.0f} m, "
              f"lingering after clear {stats['linger_m']:.0f} m")
        import json
        rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "runs")
        os.makedirs(rd, exist_ok=True)
        summ = dict(tag=args.tag, time_s=round(time.time() - t0), dist_m=round(dist), speed=args.speed,
                    collisions=sorted(stats["collisions"]), red_violations=stats["red_violations"],
                    swerves=stats["swerves"], recoveries=stats["recoveries"], xte_mean=round(float(xte.mean()), 3),
                    xte_p95=round(float(np.percentile(xte, 95)), 3), wrong_m=round(stats["wrong_m"], 1),
                    linger_m=round(stats["linger_m"], 1), excursions=exs)
        with open(os.path.join(rd, time.strftime("%Y%m%d_%H%M%S") + f"_{args.tag}.json"), "w") as f_:
            json.dump(summ, f_, indent=1)


if __name__ == "__main__":
    main()
