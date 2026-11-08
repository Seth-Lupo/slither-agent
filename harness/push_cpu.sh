#!/bin/bash
# Local: copy the harness to every node of a Slurm job (default: the dyna-rl-browsers CPU hold) and set it up.
#   ./push_cpu.sh [jobid]
set -uo pipefail
CFG=~/video-gen/.ssh_config; HERE=$(cd "$(dirname "$0")" && pwd)
J=${1:-$(ssh -F $CFG vg-login "squeue -h -u slupo01 -n dyna-rl-browsers -o %i" 2>/dev/null | head -1)}
NODES=$(ssh -F $CFG vg-login "scontrol show hostnames \$(squeue -h -j $J -o %N)" 2>/dev/null)
[ -n "$NODES" ] || { echo "job $J not running"; exit 1; }
E="ssh -F $CFG -J vg-login -o StrictHostKeyChecking=accept-new"
for N in $NODES; do (
  $E slupo01@$N "mkdir -p /tmp/slither/harness" 2>/dev/null
  rsync -a --exclude .venv --exclude __pycache__ --exclude 'episodes*' -e "$E" $HERE/ slupo01@$N:/tmp/slither/harness/ 2>/dev/null
  $E slupo01@$N "bash /tmp/slither/harness/scripts/setup_node.sh" 2>&1 | grep -vi "quantum\|store now\|upgraded" | sed "s/^/$N: /"
) & done; wait
echo "$NODES" > $HERE/.cpu_nodes
