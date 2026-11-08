#!/bin/bash
# On the GPU node (job 6956433): massive self-play on all 8 GPUs (torchrun x8), restarts on crash (resumes ckpt.pt).
# Dashboard data -> /tmp/slither/sp/dash, served at https://slither.sethlupo.com.   Stop: bash /tmp/slither/sp/code/stop.sh
R=/tmp/slither/sp
export SP_ROOT=$R HF_HOME=/tmp/slither/mg/hf_home HF_TOKEN=$(cat /tmp/slither/mg/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
mkdir -p $R/dash $R/logs; rm -f $R/STOP
# public dashboard: page + data link; (re)start web server + tunnel if they are not running
mkdir -p /tmp/slither/rl/web/sp; cp $R/code/dash/index.html /tmp/slither/rl/web/sp/index.html; ln -sfn $R/dash /tmp/slither/rl/web/sp/data
pgrep -f "[c]loudflared tunnel" > /dev/null || setsid nohup bash /tmp/slither/rl/code/web.sh > $R/logs/websh.log 2>&1 < /dev/null &
( while [ ! -f $R/STOP ]; do echo "[$(date +%F' '%T)] start train" >> $R/logs/train.log
    cd $R/code && /tmp/slither/mg/venv/bin/torchrun --standalone --nproc_per_node 8 train.py >> $R/logs/train.log 2>&1
    echo "[$(date +%F' '%T)] train exited $?" >> $R/logs/train.log; sleep 20; done ) &
( while [ ! -f $R/STOP ]; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' 'BEGIN{printf "{\"t\":'$(date +%s)',\"gpus\":["} {printf "%s{\"i\":%s,\"util\":%s,\"mem\":%s}", (NR>1?",":""), $1,$2,$3} END{printf "]}"}' > $R/dash/sys.json.tmp
    mv $R/dash/sys.json.tmp $R/dash/sys.json
    pgrep -f "[c]loudflared tunnel" > /dev/null || setsid nohup bash /tmp/slither/rl/code/web.sh >> $R/logs/websh.log 2>&1 < /dev/null &
    sleep 20; done ) &
echo "started self-play: logs in $R/logs"
wait
