"""Table of logs/runs/*.json grouped by planner tag."""
import glob, json, os, collections
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/runs/*.json")))]
g = collections.defaultdict(list)
for r in runs:
    g[r["tag"]].append(r)
print(f"{'tag':6} {'laps':>4} {'km':>5} {'coll':>4} {'red':>3} {'wrong-lane %':>12} {'linger m/lap':>12} {'max exc m':>9} {'xte mean':>8}")
for t, rs in g.items():
    d = sum(r["dist_m"] for r in rs)
    print(f"{t:6} {len(rs):4} {d/1000:5.2f} {sum(len(r['collisions']) for r in rs):4} {sum(r['red_violations'] for r in rs):3} "
          f"{100*sum(r['wrong_m'] for r in rs)/max(d,1):12.1f} {sum(r['linger_m'] for r in rs)/len(rs):12.0f} "
          f"{max([e['len_m'] for r in rs for e in r['excursions']], default=0):9.0f} "
          f"{sum(r['xte_mean'] for r in rs)/len(rs):8.2f}")
