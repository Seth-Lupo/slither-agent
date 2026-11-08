#!/bin/bash
# On the GPU node: the self-play policy plays (and keeps learning in) 80 live slither.io games.
#   live.py actors: 10 processes x 2 proxies x 4 games, GPUs 0-7, one decision per 0.1 s   GPU 3: learn.py (PPO)
# Self-play keeps running alongside. Web: https://slither.sethlupo.com  (/live/ and /sp/).   Stop: bash /tmp/slither/live/code/stop.sh
R=/tmp/slither/live; PY=/tmp/slither/mg/venv/bin/python
export LIVE_ROOT=$R SP_CODE=/tmp/slither/sp/code HARNESS=/tmp/slither/harness PLAYWRIGHT_BROWSERS_PATH=/tmp/slither/harness/browsers
export SLITHER_RENDER=nogpu SLITHER_DEBUG_DIR=$R/dash/debug HF_HOME=/tmp/slither/mg/hf_home HF_TOKEN=$(cat /tmp/slither/mg/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
mkdir -p $R/frag $R/dash/s $R/logs; rm -f $R/STOP
[ -f $R/init_policy.pt ] || cp /tmp/slither/sp/policy.pt $R/init_policy.pt
# v3 (previous live agent) baseline for the dashboard
[ -f $R/dash/baseline.json ] || $PY -c "
import json,numpy as np
E=[json.loads(l) for l in open('/tmp/slither/rl2/dash/episodes.jsonl') if l.strip()]; E=[e for e in E if e.get('died')][-2000:]
f=lambda g:[e for e in E if bool(e.get('greedy'))==g]
o={k:dict(n=len(f(g)),surv_med=float(np.median([e['seconds'] for e in f(g)])),best_med=float(np.median([e['best'] for e in f(g)]))) for k,g in (('sample',False),('greedy',True))}
json.dump(o,open('$R/dash/baseline.json','w'));print(o)"
W=/tmp/slither/rl/web; mkdir -p $W/sp $W/live
cp $R/code/dash/home.html $W/index.html; cp $R/code/dash/index.html $W/live/index.html; ln -sfn $R/dash $W/live/data
cp /tmp/slither/sp/code/dash/index.html $W/sp/index.html; ln -sfn /tmp/slither/sp/dash $W/sp/data
keep() { local name=$1 gpu=$2; shift 2
  ( while [ ! -f $R/STOP ]; do echo "[$(date +%F' '%T)] start $name" >> $R/logs/$name.log
      CUDA_VISIBLE_DEVICES=$gpu "$@" >> $R/logs/$name.log 2>&1; echo "[$(date +%F' '%T)] $name exited $?" >> $R/logs/$name.log; sleep 15; done ) & }
keep learner 3 $PY $R/code/learn.py
for r in $(seq 0 9); do keep actor$r $((r % 8)) $PY $R/code/live.py --rank $r --proxy_lo $((2 * r)) --proxy_hi $((2 * r + 2)); done   # 10 processes x 8 games (CPU-bound per process)
echo "started live: logs in $R/logs"
wait
