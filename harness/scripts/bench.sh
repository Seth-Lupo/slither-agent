#!/bin/bash
# On a CPU node: stop any harness run, then start N sessions for S seconds in the background (survives ssh/VPN drops).
#   bash scripts/bench.sh [sessions=64] [seconds=240]     -> /tmp/slither/bench.log, /tmp/slither/load.log
N=${1:-64}; S=${2:-240}; R=/tmp/slither/harness
pkill -u $USER -f "scripts/run.py" ; pkill -u $USER -f "chrome-headless-shell"; pkill -u $USER -f "loadmon"; sleep 2
rm -rf /tmp/slither/bench_eps
cd $R; export PLAYWRIGHT_BROWSERS_PATH=$R/browsers SLITHER_RENDER=${SLITHER_RENDER:-nogpu}
setsid nohup .venv/bin/python scripts/run.py --sessions $N --seconds $S --render_fps 20 --stagger 0.5 --out /tmp/slither/bench_eps > /tmp/slither/bench.log 2>&1 < /dev/null &
setsid nohup bash -c 'exec -a loadmon bash -c "while sleep 20; do echo \$(date +%T) \$(cut -d\" \" -f1 /proc/loadavg); done"' > /tmp/slither/load.log 2>&1 < /dev/null &
echo "$(hostname): started $N sessions for ${S}s"
