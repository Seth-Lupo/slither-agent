#!/bin/bash
# Stop the Dyna-Q system (keeps all data in /tmp/slither/rl).
touch /tmp/slither/rl/STOP; pkill -f "/tmp/slither/rl/code/" ; pkill -f "[c]hrome-headless-shell"; echo stopped
