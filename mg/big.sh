#!/bin/bash
# H200 job: s400 (400M dynamics) on all GPUs of the node. Data, tokenizer and earlier runs come from the HF backup,
# so nothing is re-encoded. Restarts resume from the run's checkpoint; the dashboard keeps s150/s50 for comparison.
set -uo pipefail
R=/tmp/slither/mg; cd $R/code
export MG_ROOT=$R HF_HOME=$R/hf_home HF_TOKEN=$(cat $R/.hf_token) HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 UV_CACHE_DIR=$R/uvcache
export MG_HF_SKIP=s150,s150x,s150ev,pilot50 MG_HF_EVERY=1800   # RTX job owns s150/s150x (stale here); latest every 30 min
source /etc/profile.d/modules.sh 2>/dev/null; module load uv/0.9.18
NG=$(nvidia-smi -L | wc -l); log() { echo "[$(date +%F' '%T)] $*"; }
log "node $(hostname), $NG GPUs, $(nproc) CPUs"
if [ ! -x $R/venv/bin/python ]; then
  log "venv"; uv venv -q $R/venv --python 3.11 && uv pip install -q -p $R/venv/bin/python "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 && uv pip install -q -p $R/venv/bin/python numpy pillow huggingface_hub && rm -rf $R/uvcache
fi
PY=$R/venv/bin/python
[ -f $R/data/tokens.npy ] || { log "restore"; $PY hf.py restore || exit 1; }
# 2026-10-18: s400 ended by request; follow-on jobs only hold the H200s (data + venv ready, GPUs free). Delete this line to resume s400.
log "holding node: s400 ended, GPUs free"; exec sleep infinity
$PY hf.py loop > $R/backup.log 2>&1 &
for i in 1 2 3; do [ -f $R/runs/s400/DONE ] && break; log "dynamics ($i): s400 on $NG GPUs"
  $R/venv/bin/torchrun --standalone --nproc_per_node $NG dynamics.py --name s400 --size 400 --bs ${S400_BS:-32} >> $R/s400.log 2>&1; done
$PY hf.py backup; log "all done"

# --- continuation (appended 2026-10-17): once s400's 4 epochs are DONE, keep training as s400x indefinitely. Each Slurm job
# re-warms and fits a fresh cosine to its walltime (--cycle); `ctl.sh keep` queues the next job when this one ends.
if [ -f $R/runs/s400/DONE ]; then
  for i in 1 2 3; do log "dynamics ($i): s400x continuation on $NG GPUs"
    $R/venv/bin/torchrun --standalone --nproc_per_node $NG dynamics.py --name s400x --size 400 --bs ${S400_BS:-32} \
      --init $R/runs/s400/ckpt.pt --epochs 1000 --cycle 1 >> $R/s400x.log 2>&1; done
  $PY hf.py backup; log "s400x: walltime used up"
fi
