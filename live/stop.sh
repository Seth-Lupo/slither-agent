#!/bin/bash
# Stop live play (data stays in /tmp/slither/live; self-play and the web page keep running).
touch /tmp/slither/live/STOP; for p in $(pgrep -f "[l]ive/code/(launch.sh|live.py|learn.py)"); do kill $p; done
for p in $(pgrep -u $USER -f "[c]hrome-headless-shell"); do kill $p 2>/dev/null; done; echo stopped
