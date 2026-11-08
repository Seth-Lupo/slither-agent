#!/bin/bash
# On the GPU node (job 6956433): start the whole Dyna-Q system. Everything in node-local /tmp, backups to the HF bucket.
#   GPU 0: actor.py   (80 live slither.io games via 20 proxies x 4, acting with the latest Q head)
#   GPU 1: learner.py (Q / reward / death heads on frozen s150p features; real + imagined replay)
#   GPU 2-7: imagine.py (world-model rollouts of real states under random actions -> imagined transitions)
# Each process is restarted if it crashes. Stop everything: bash /tmp/slither/rl/code/stop.sh
R=/tmp/slither/rl; PY=/tmp/slither/mg/venv/bin/python
export MG_ROOT=/tmp/slither/mg RL_ROOT=$R HARNESS=/tmp/slither/harness PLAYWRIGHT_BROWSERS_PATH=/tmp/slither/harness/browsers
export SLITHER_RENDER=nogpu HF_HOME=/tmp/slither/mg/hf_home HF_TOKEN=$(cat /tmp/slither/mg/.hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
mkdir -p $R/real $R/imag $R/dash $R/logs
keep() { local name=$1 gpu=$2; shift 2
  ( while [ ! -f $R/STOP ]; do echo "[$(date +%F' '%T)] start $name" >> $R/logs/$name.log
      CUDA_VISIBLE_DEVICES=$gpu "$@" >> $R/logs/$name.log 2>&1; echo "[$(date +%F' '%T)] $name exited $?" >> $R/logs/$name.log; sleep 15; done ) & }
rm -f $R/STOP
keep learner 1 $PY $R/code/learner.py
keep actor 0 $PY $R/code/actor.py --proxies /tmp/slither/proxies.txt --per_proxy ${PER_PROXY:-4} --sessions ${SESSIONS:-80}
for g in 2 3 4 5 6 7; do keep imag$g $g $PY $R/code/imagine.py --rank $g; done
( while [ ! -f $R/STOP ]; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F', ' 'BEGIN{printf "{\"t\":'$(date +%s)',\"gpus\":["} {printf "%s{\"i\":%s,\"util\":%s,\"mem\":%s}", (NR>1?",":""), $1,$2,$3} END{printf "],"}' > $R/dash/sys.json.tmp
    echo "\"load\":$(cut -d' ' -f1 /proc/loadavg),\"cpus\":$(nproc),\"mem_used_g\":$(free -g | awk '/Mem/{print $3}'),\"tmp_used_g\":$(df -BG /tmp | awk 'NR==2{print int($3)}'),\"real_files\":$(ls $R/real | wc -l),\"real_mb\":$(du -sm $R/real | cut -f1)}" >> $R/dash/sys.json.tmp
    mv $R/dash/sys.json.tmp $R/dash/sys.json; sleep 20; done ) &
echo "started: logs in $R/logs"
wait
