#!/bin/bash
# Full AV demo: traffic lights + scenario (people, crossers, lane obstacles) + AV stack v2.
# Needs the sim running in car mode (./run_sim.sh). Usage: ./run_av_demo.sh [--laps N]
cd "$(dirname "$0")"
export DISPLAY=:0 XAUTHORITY=$(ls /run/user/1000/.mutter-Xwaylandauth.* | head -1)
.venv/bin/python -c "import airsim,time; c=airsim.CarClient(); c.reset(); time.sleep(1)" >/dev/null 2>&1
.venv/bin/python -u signals.py & SIG=$!
.venv/bin/python -u scenario.py & SCN=$!
sleep 12
trap "kill -TERM $SIG $SCN" EXIT
.venv/bin/python -u av_stack.py "$@"
kill -TERM $SIG $SCN; wait $SIG $SCN 2>/dev/null
