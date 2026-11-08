#!/bin/bash
# Stop RL v2 (keeps data in /tmp/slither/rl2; the public dashboard keeps running).
touch /tmp/slither/rl2/STOP; for p in $(pgrep -f "[r]l2/code/(launch.sh|actor.py|learner.py)"); do kill $p; done
for p in $(pgrep -u $USER -f "[c]hrome-headless-shell"); do kill $p 2>/dev/null; done; echo stopped
