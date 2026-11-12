#!/bin/bash
# Repeated AV laps for measuring error rates. Sim must be running (./run_sim.sh).
# Usage: ./eval_loop.sh PLANNER N [extra av_stack args]   e.g. ./eval_loop.sh v3 3 --speed 13.9
cd "$(dirname "$0")"
PL=${1:-v3}; N=${2:-3}; shift 2
for i in $(seq 1 "$N"); do
  echo "=== $PL lap $i/$N  $(date +%T)"
  ./run_av_demo.sh --laps 1 --planner "$PL" --tag "$PL" --stream-file /dev/shm/av_dash.jpg "$@" 2>&1 \
    | tr '\r' '\n' | grep -E "====|collisions|red-light|traffic light|swerves|lateral|oncoming|time "
done
.venv/bin/python summarize_runs.py
