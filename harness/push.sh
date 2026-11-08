#!/bin/bash
# Local: copy the harness to the node of a running Slurm job and set it up there.
#   ./push.sh [jobid]      (default: the dyna-rl-slither hold, 6956433)
set -euo pipefail
CFG=~/video-gen/.ssh_config; J=${1:-6956433}; HERE=$(cd "$(dirname "$0")" && pwd)
N=$(ssh -F $CFG vg-login "squeue -h -j $J -o %N" 2>/dev/null); [ -n "$N" ] || { echo "job $J is not running yet"; exit 1; }
E="ssh -F $CFG -J vg-login"
$E slupo01@$N "mkdir -p /tmp/slither/harness"
rsync -a --exclude .venv --exclude __pycache__ --exclude 'episodes*' -e "$E" $HERE/ slupo01@$N:/tmp/slither/harness/
$E slupo01@$N "bash /tmp/slither/harness/scripts/setup_node.sh"
echo "on $N: cd /tmp/slither/harness && PLAYWRIGHT_BROWSERS_PATH=/tmp/slither/harness/browsers .venv/bin/python scripts/run.py --sessions 8 --seconds 120 --render_fps 20"
