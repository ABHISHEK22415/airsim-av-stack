#!/usr/bin/env python3
"""Autonomous car in AirSimNH: plan a route on the street grid, follow the right-hand lane and obey
virtual traffic signals at every 3/4-way junction.

Start the sim in car mode first:   ./run_sim.sh &        (uses settings_car.json)
Then:
    .venv/bin/python drive_to.py                      # default goal: far corner (128, 127)
    .venv/bin/python drive_to.py --goal -128 127
    .venv/bin/python drive_to.py --goal 60 -129 --speed 10

World frame is AirSim NED (x north, y east, metres), origin = the car's start (centre of junction (0,-1)).
Traffic lights are drawn in the sim as coloured lamps above each junction approach; they cycle
GREEN 12 s -> YELLOW 3 s -> ALL-RED 2 s per axis (north-south vs east-west), offset per junction.
"""
import argparse
import heapq
import math
import time

import airsim

# ---------------- road graph (centre lines, measured from Road_* tiles) ----------------
XS = [-128, 0, 128]
YS = [-129, -1, 127]
NODES = [(x, y) for x in XS for y in YS] + [(80, -129), (80, -1)]
EDGES = set()
for y in YS:  # east-west streets run along x at constant y
    xs = sorted({x for x, yy in NODES if yy == y})
    EDGES |= {((a, y), (b, y)) for a, b in zip(xs, xs[1:])}
for x in XS + [80]:  # streets at constant x
    ys = sorted({y for xx, y in NODES if xx == x})
    if x == 80:
        ys = [-129, -1]  # x=80 street only exists between y=-129 and y=-1
    EDGES |= {((x, a), (x, b)) for a, b in zip(ys, ys[1:])}
ADJ = {n: [] for n in NODES}
for a, b in EDGES:
    ADJ[a].append(b)
    ADJ[b].append(a)
SIGNALS = [n for n in NODES if len(ADJ[n]) >= 3]

LANE = 1.9          # metres right of the centre line
JUNCTION_R = 8.0    # half size of a junction box
STOP_BACK = 11.0    # stop line distance before junction centre

GREEN, YELLOW, ALLRED = 12.0, 3.0, 2.0
CYCLE = 2 * (GREEN + YELLOW + ALLRED)


def light(node, axis, t):
    """axis 'x' = travelling along x (north/south), 'y' = along y. Returns 'G', 'Y' or 'R'."""
    ph = (t + (node[0] * 0.07 + node[1] * 0.11) % CYCLE) % CYCLE
    if axis == "y":
        ph = (ph + CYCLE / 2) % CYCLE
    if ph < GREEN:
        return "G"
    if ph < GREEN + YELLOW:
        return "Y"
    return "R"


def unit(a, b):
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d)


def right(d):
    return (-d[1], d[0])  # NED: right of north is east


def add(p, q, k=1.0):
    return (p[0] + q[0] * k, p[1] + q[1] * k)


def dijkstra(start, goal, heading):
    pq, seen = [(0, start, [start])], {}
    while pq:
        cost, n, path = heapq.heappop(pq)
        if n == goal:
            return path
        if n in seen:
            continue
        seen[n] = cost
        for m in ADJ[n]:
            d = unit(n, m)
            if n == start and d[0] * heading[0] + d[1] * heading[1] < -0.5:
                continue  # no U-turn at the start
            heapq.heappush(pq, (cost + math.dist(n, m), m, path + [m]))
    return None


def bezier(p0, p1, p2, n=10):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in [i / n for i in range(n + 1)]]


def build_path(start_pos, route, goal_pt):
    """Dense lane path + list of stop lines (s, node, axis)."""
    pts = [start_pos]
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        d = unit(a, b)
        r = right(d)
        if i > 0:  # turn / go straight through junction a
            dprev = unit(route[i - 1], a)
            rp = right(dprev)
            p0 = add(add(a, dprev, -JUNCTION_R), rp, LANE)
            p2 = add(add(a, d, JUNCTION_R), r, LANE)
            ctrl = add(add(a, rp, LANE), r, LANE)
            if abs(dprev[0] * d[0] + dprev[1] * d[1]) > 0.9:  # straight
                ctrl = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)
            pts += bezier(p0, ctrl, p2)
        else:
            pts.append(add(add(a, d, JUNCTION_R), r, LANE))
        end = add(add(b, d, -JUNCTION_R), r, LANE) if i < len(route) - 2 else add(goal_pt, r, LANE)
        pts.append(end)
    # resample every 1 m
    dense = [pts[0]]
    for p in pts[1:]:
        q = dense[-1]
        L = math.dist(p, q)
        for k in range(1, int(L) + 1):
            dense.append((q[0] + (p[0] - q[0]) * k / L, q[1] + (p[1] - q[1]) * k / L))
        if L % 1 > 0.05:
            dense.append(p)
    s = [0.0]
    for i in range(1, len(dense)):
        s.append(s[-1] + math.dist(dense[i - 1], dense[i]))
    stops = []
    jprev = 0
    for i in range(1, len(route) - 1):
        n = route[i]
        if n in SIGNALS:
            d = unit(route[i - 1], n)
            sp = add(add(n, d, -STOP_BACK), right(d), LANE)
            j = next((k for k in range(jprev, len(dense)) if math.dist(dense[k], sp) < 2.5),
                     min(range(jprev, len(dense)), key=lambda k: math.dist(dense[k], sp)))
            jprev = j + 1
            stops.append([s[j], n, "x" if abs(d[0]) > 0.5 else "y", None])
    return dense, s, stops


def draw_static(c, path):
    c.simFlushPersistentMarkers()
    c.simPlotLineStrip([airsim.Vector3r(x, y, -0.25) for x, y in path], color_rgba=[0.1, 0.8, 1.0, 1.0],
                       thickness=12, is_persistent=True)
    gx, gy = path[-1]
    c.simPlotLineList([airsim.Vector3r(gx, gy, 0), airsim.Vector3r(gx, gy, -6)], color_rgba=[0.2, 1, 0.2, 1],
                      thickness=25, is_persistent=True)
    c.simPlotStrings(["GOAL"], [airsim.Vector3r(gx, gy, -7)], scale=6, color_rgba=[0.2, 1, 0.2, 1], duration=3600)
    poles = []
    for n in SIGNALS:
        for m in ADJ[n]:
            d = unit(m, n)  # approach direction into n
            p = add(add(n, d, -STOP_BACK + 1), right(d), 6.0)
            poles += [airsim.Vector3r(p[0], p[1], 0), airsim.Vector3r(p[0], p[1], -4.2)]
    c.simPlotLineList(poles, color_rgba=[0.25, 0.25, 0.25, 1], thickness=6, is_persistent=True)


COL = {"G": [0.1, 1, 0.2, 1], "Y": [1, 0.8, 0, 1], "R": [1, 0.1, 0.1, 1]}


def draw_lights(c, t):
    pts = {"G": [], "Y": [], "R": []}
    for n in SIGNALS:
        for m in ADJ[n]:
            d = unit(m, n)
            p = add(add(n, d, -STOP_BACK + 1), right(d), 6.0)
            st = light(n, "x" if abs(d[0]) > 0.5 else "y", t)
            pts[st].append(airsim.Vector3r(p[0], p[1], -4.6))
    for st, ps in pts.items():
        if ps:
            c.simPlotPoints(ps, color_rgba=COL[st], size=28, duration=0.35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", nargs=2, type=float, default=[128, 127], metavar=("X", "Y"))
    ap.add_argument("--speed", type=float, default=9.0, help="cruise speed m/s")
    ap.add_argument("--tour", type=int, default=0, metavar="LAPS",
                    help="drive the city loop LAPS times (starting from the centre junction) instead of --goal")
    args = ap.parse_args()

    c = airsim.CarClient()
    c.confirmConnection()
    c.enableApiControl(True)
    ctl = airsim.CarControls()

    st = c.getCarState().kinematics_estimated
    pos = (st.position.x_val, st.position.y_val)
    yaw = airsim.to_eularian_angles(st.orientation)[2]
    heading = (math.cos(yaw), math.sin(yaw))

    # snap goal to the nearest street edge
    def proj(p, a, b):
        d = unit(a, b)
        L = math.dist(a, b)
        t = max(JUNCTION_R + 4, min(L - JUNCTION_R - 4, (p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]))
        return add(a, d, t), t
    goal = tuple(args.goal)
    # where does the route begin? at a junction if we are inside one, else at the junction ahead on our street
    near = min(NODES, key=lambda n: math.dist(n, pos))
    if math.dist(near, pos) < JUNCTION_R + 2:
        prefix, start, h0 = [], near, heading
    else:
        def seg_dist(a, b):
            d = unit(a, b)
            t = max(0, min(math.dist(a, b), (pos[0] - a[0]) * d[0] + (pos[1] - a[1]) * d[1]))
            return math.dist(add(a, d, t), pos), add(a, d, t)
        a, b = min(EDGES, key=lambda e: seg_dist(*e)[0])
        if (b[0] - a[0]) * heading[0] + (b[1] - a[1]) * heading[1] < 0:
            a, b = b, a
        prefix, start, h0 = [seg_dist(a, b)[1]], b, unit(a, b)
    best = None
    for a, b in EDGES:
        for u, v in [(a, b), (b, a)]:
            gp, t = proj(goal, u, v)
            route = dijkstra(start, u, h0)
            if not route:
                continue
            cost = sum(math.dist(route[i], route[i + 1]) for i in range(len(route) - 1)) + t + math.dist(gp, goal) * 50
            if best is None or cost < best[0]:
                best = (cost, prefix + route + [v], gp)
    _, route, gpt = best
    if args.tour:
        # city loop through all districts; consecutive points are adjacent junctions
        loop = [(0, -1), (80, -1), (128, -1), (128, 127), (0, 127), (-128, 127), (-128, -1),
                (-128, -129), (0, -129), (0, -1)]
        start_n = min(NODES, key=lambda n: math.dist(n, pos))
        lead = dijkstra(start_n, loop[0], heading) if start_n != loop[0] else [loop[0]]
        route = lead[:-1] + loop[:-1] * args.tour + [(0, -1), (80, -1)]
        gpt = (40, -1)
    path, S, stops = build_path(pos, route, gpt)
    print(f"Route via junctions {route[:-1]} -> goal {tuple(round(v) for v in gpt)}, {S[-1]:.0f} m, "
          f"{len(stops)} traffic signals")
    draw_static(c, path)

    idx, t0, last_draw, stuck_since = 0, time.time(), 0, None
    last_col, last_obj = None, None
    try:
        while True:
            now = time.time()
            t = now - t0
            if now - last_draw > 0.3:
                draw_lights(c, now)
                last_draw = now
            cs = c.getCarState()
            k = cs.kinematics_estimated
            x, y = k.position.x_val, k.position.y_val
            yaw = airsim.to_eularian_angles(k.orientation)[2]
            v = cs.speed
            # progress along path
            idx = min(range(idx, min(idx + 40, len(path))), key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2)
            s_cur = S[idx]
            # pure pursuit
            Ld = 5.0 + 0.45 * v
            j = idx
            while j < len(path) - 1 and S[j] - s_cur < Ld:
                j += 1
            tx, ty = path[j]
            alpha = math.atan2(ty - y, tx - x) - yaw
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            delta = math.atan2(2 * 2.8 * math.sin(alpha), max(Ld, 1e-3))
            ctl.steering = max(-1.0, min(1.0, delta / 0.55))
            # speed planning
            vt = args.speed
            # slow for curves ahead (heading change over next 25 m)
            j2 = idx
            while j2 < len(path) - 2 and S[j2] - s_cur < 25:
                j2 += 1
            if j2 > idx + 2:
                h1 = math.atan2(path[idx + 2][1] - path[idx][1], path[idx + 2][0] - path[idx][0])
                h2 = math.atan2(path[j2][1] - path[j2 - 2][1], path[j2][0] - path[j2 - 2][0])
                turn = abs(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1)))
                if turn > 0.5:
                    vt = min(vt, 4.5)
            # traffic lights
            status = "no signal ahead"
            for sl in stops:
                d = sl[0] - s_cur
                if d < -1:
                    continue
                state = light(sl[1], sl[2], now)
                name = {"G": "GREEN", "Y": "YELLOW", "R": "RED"}[state]
                status = f"signal {sl[1]} {name} in {d:4.0f} m"
                must_stop = state == "R" or (state == "Y" and sl[3] != "go")
                if state == "Y" and sl[3] is None:  # decide once on yellow
                    sl[3] = "go" if d < v * v / (2 * 3.0) + 1 else "stop"
                    must_stop = sl[3] == "stop"
                if state == "G":
                    sl[3] = None
                if must_stop:
                    vt = min(vt, math.sqrt(max(0.0, 2 * 2.5 * (d - 1.0))))
                break
            # goal
            d_goal = S[-1] - s_cur
            vt = min(vt, math.sqrt(max(0.0, 2 * 2.0 * (d_goal - 0.5))))
            # longitudinal control
            err = vt - v
            if vt < 0.3 and v < 1.5:
                ctl.throttle, ctl.brake = 0.0, 1.0
            elif err > 0:
                ctl.throttle, ctl.brake = max(0.0, min(1.0, 0.3 + 0.12 * err)), 0.0
            elif err < -0.6:
                ctl.throttle, ctl.brake = 0.0, max(0.0, min(1.0, -0.25 * err))
            else:
                ctl.throttle, ctl.brake = 0.15, 0.0
            # stuck recovery: wanted to move but not moving for 3 s -> reverse 2 s with opposite steer
            if vt > 2 and v < 0.3:
                stuck_since = stuck_since or now
                if now - stuck_since > 3:
                    print("\n[stuck - reversing]")
                    c.setCarControls(airsim.CarControls(throttle=-0.5, steering=-ctl.steering, is_manual_gear=True,
                                                        manual_gear=-1))
                    time.sleep(2.0)
                    c.setCarControls(airsim.CarControls(brake=1))
                    time.sleep(0.5)
                    ctl = airsim.CarControls()
                    stuck_since = None
                    continue
            else:
                stuck_since = None
            c.setCarControls(ctl)
            c.simPrintLogMessage("Car: ", f"{v*3.6:5.1f} km/h | {status} | goal in {d_goal:5.0f} m", 0)
            print(f"\r t={t:5.1f}s v={v*3.6:5.1f} km/h  {status:34s} goal {d_goal:5.0f} m", end="", flush=True)
            if d_goal < 1.5 and v < 0.3:
                print("\nArrived at goal.")
                break
            ci = c.simGetCollisionInfo()
            if ci.has_collided and ci.time_stamp != last_col and t > 2:
                last_col = ci.time_stamp
                if ci.object_name != last_obj:
                    print(f"\n[collision with {ci.object_name}]")
                    last_obj = ci.object_name
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ctl.throttle, ctl.brake, ctl.steering = 0, 1, 0
        c.setCarControls(ctl)
        c.enableApiControl(False)


if __name__ == "__main__":
    main()
