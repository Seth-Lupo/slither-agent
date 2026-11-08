#!/bin/bash
# Stop self-play (keeps ckpt.pt and dash/; the public dashboard keeps running).
touch /tmp/slither/sp/STOP; for p in $(pgrep -f "[s]p/code/launch.sh|[t]rain.py"); do kill $p; done; echo stopped
