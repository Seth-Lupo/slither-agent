#!/bin/bash
# Local: Dyna-Q dashboard at http://localhost:8773, synced from the GPU node every 30 s.
# Goes through Slurm (srun --overlap) because direct ssh to the node can hang while 80 browsers run.
CFG=~/video-gen/.ssh_config; HERE=$(cd "$(dirname "$0")" && pwd); J=${JOB:-6956433}; mkdir -p $HERE/dash/data
pkill -f "http.server 8773" 2>/dev/null; (cd $HERE/dash && python3 -m http.server 8773 --bind 127.0.0.1 >/dev/null 2>&1 &)
echo "dashboard: http://localhost:8773"
while true; do
  timeout 100 ssh -F $CFG -o ConnectTimeout=20 -o ServerAliveInterval=15 vg-login \
    "timeout 80 srun --jobid=$J --overlap -N1 -n1 tar cf - -C /tmp/slither/rl/dash --exclude='*.tmp*' ." 2>/dev/null > $HERE/dash/.sync.tar \
    && [ -s $HERE/dash/.sync.tar ] && tar xf $HERE/dash/.sync.tar -C $HERE/dash/data 2>/dev/null
  sleep 30
done
