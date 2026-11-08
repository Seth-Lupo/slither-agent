#!/bin/bash
# On the GPU node: serve the dashboard publicly at https://slither.sethlupo.com via a Cloudflare tunnel (outbound only).
# Page = $R/code/dash/index.html; data/ = the live $R/dash. Restarts each piece if it dies.
R=${RL_ROOT:-/tmp/slither/rl2}; W=/tmp/slither/web; CF=/tmp/slither/rl/cloudflared; mkdir -p $W $R/logs
cp $R/code/dash/index.html $W/index.html; ln -sfn $R/dash $W/data
[ -x $CF ] || { curl -sL -o $CF https://github.com/cloudflare/cloudflared/releases/download/2025.6.1/cloudflared-linux-amd64 && chmod +x $CF; }
( while true; do cd $W && /tmp/slither/mg/venv/bin/python -m http.server 8773 --bind 127.0.0.1 >> $R/logs/web.log 2>&1; sleep 5; done ) &
( while true; do $CF tunnel --config $R/code/cf/config.yml run >> $R/logs/tunnel.log 2>&1; sleep 10; done ) &
wait
