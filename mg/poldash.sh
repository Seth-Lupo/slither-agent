#!/bin/bash
# Local: dashboard for the s150p run at http://localhost:8772 (synced from the node every 60 s).
CFG=~/video-gen/.ssh_config; HERE=$(cd "$(dirname "$0")" && pwd); N=$(cut -d' ' -f2 $HERE/.job.pol)
(cd $HERE/dash_pol && python3 -m http.server 8772 --bind 127.0.0.1 >/dev/null 2>&1 &)
echo "dashboard: http://localhost:8772"
E="ssh -F $CFG -J vg-login -o ConnectTimeout=20 -o ServerAliveInterval=15"
while true; do
  timeout 120 rsync -a -e "$E" slupo01@$N:/tmp/slither/mg/dash/dyn/s150p/ $HERE/dash_pol/s150p/ 2>/dev/null
  timeout 60 rsync -a -e "$E" slupo01@$N:/tmp/slither/mg/dash/status.json $HERE/dash_pol/status.json 2>/dev/null
  sleep 60
done
