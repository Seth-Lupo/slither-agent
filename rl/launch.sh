#!/bin/bash
# On the GPU node (job 6956433): RL v2 = PPO fine-tuning of the pretrained s150p policy on 80 live games.
#   GPU 0-2: actor.py (proxies 0-6 / 7-13 / 14-19, 4 games each, one decision per 0.1 s)
#   GPU 3-7: learner.py (torchrun x5: PPO + world-model + danger foresight, top backbone blocks fine-tuned)
# Everything in /tmp/slither/rl2; restarts crashed processes. Stop: bash /tmp/slither/rl2/code/stop.sh
R=/tmp/slither/rl2; PY=/tmp/slither/mg/venv/bin/python
export MG_ROOT=/tmp/slither/mg RL_ROOT=$R HARNESS=/tmp/slither/harness PLAYWRIGHT_BROWSERS_PATH=/tmp/slither/harness/browsers
export SLITHER_RENDER=nogpu SLITHER_DEBUG_DIR=$R/dash/debug HF_HOME=/tmp/slither/mg/hf_home HF_TOKEN=$(cat /tmp/slither/mg/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
mkdir -p $R/frag $R/dash/s $R/logs; rm -f $R/STOP
keep() { local name=$1 gpu=$2; shift 2
  ( while [ ! -f $R/STOP ]; do echo "[$(date +%F' '%T)] start $name" >> $R/logs/$name.log
      CUDA_VISIBLE_DEVICES=$gpu "$@" >> $R/logs/$name.log 2>&1; echo "[$(date +%F' '%T)] $name exited $?" >> $R/logs/$name.log; sleep 15; done ) & }
keep learner 3,4,5,6,7 /tmp/slither/mg/venv/bin/torchrun --standalone --nproc_per_node 5 $R/code/learner.py
keep actor0 0 $PY $R/code/actor.py --rank 0 --proxy_lo 0 --proxy_hi 7
keep actor1 1 $PY $R/code/actor.py --rank 1 --proxy_lo 7 --proxy_hi 14
keep actor2 2 $PY $R/code/actor.py --rank 2 --proxy_lo 14 --proxy_hi 20
( while [ ! -f $R/STOP ]; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' 'BEGIN{printf "{\"t\":'$(date +%s)',\"gpus\":["} {printf "%s{\"i\":%s,\"util\":%s,\"mem\":%s}", (NR>1?",":""), $1,$2,$3} END{printf "],"}' > $R/dash/sys.json.tmp
    echo "\"load\":$(cut -d' ' -f1 /proc/loadavg),\"cpus\":$(nproc),\"mem_used_g\":$(free -g | awk '/Mem/{print $3}'),\"tmp_used_g\":$(df -BG /tmp | awk 'NR==2{print int($3)}')}" >> $R/dash/sys.json.tmp
    mv $R/dash/sys.json.tmp $R/dash/sys.json; sleep 20; done ) &
echo "started v2: logs in $R/logs"
wait
