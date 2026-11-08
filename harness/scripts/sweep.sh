#!/bin/bash
# On a node: how many concurrent live games does this IP get? Runs n sessions for 100 s for each n, logs a summary line.
R=/tmp/slither/harness; cd $R; export PLAYWRIGHT_BROWSERS_PATH=$R/browsers SLITHER_RENDER=nogpu
for n in "$@"; do
  pkill -u $USER -f "chrome-headless-shell"; sleep 20
  .venv/bin/python scripts/run.py --sessions $n --seconds 100 --render_fps 20 --stagger 1 --out /tmp/slither/sweep_$n > /tmp/slither/sweep_$n.log 2>&1
  echo "n=$n $(grep decisions/s /tmp/slither/sweep_$n.log | tail -1 | cut -c20-60) | no-start $(grep -c 'did not start' /tmp/slither/sweep_$n.log)"
done
