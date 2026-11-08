#!/bin/bash
# Second job: A/B of the death-event loss. s150ev starts from the main run's latest s150 checkpoint (from the HF backup)
# with the same schedule; the main job's s150 keeps plain CE as the baseline. Backs up only its own run and dashboard.
set -uo pipefail
R=/tmp/slither/mg; cd $R/code
export MG_ROOT=$R HF_TOKEN=$(cat $R/.hf_token) HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 UV_CACHE_DIR=$R/uvcache MG_HF_DASH=dash_ev
source /etc/profile.d/modules.sh 2>/dev/null; module load uv/0.9.18
NG=$(nvidia-smi -L | wc -l); log() { echo "[$(date +%F' '%T)] $*"; }
log "node $(hostname), $NG GPUs, $(nproc) CPUs"
if [ ! -x $R/venv/bin/python ]; then
  log "venv"; uv venv -q $R/venv --python 3.11 && uv pip install -q -p $R/venv/bin/python "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 && uv pip install -q -p $R/venv/bin/python numpy pillow huggingface_hub && rm -rf $R/uvcache
fi
PY=$R/venv/bin/python
if [ ! -f $R/init_s150.pt ]; then
  log "restore"; $PY hf.py restore || exit 1
  mv $R/runs/s150/ckpt.pt $R/init_s150.pt && rm -rf $R/runs/s150 $R/runs/pilot50 $R/dash && mkdir -p $R/dash || exit 1
fi
[ -f $R/data/deaths.npz ] || { log "events"; $PY events.py || exit 1; }
$PY hf.py loop > $R/backup.log 2>&1 &
for i in 1 2 3; do [ -f $R/runs/s150ev/DONE ] && break; log "dynamics ($i): s150ev on $NG GPUs"   # restarts resume from the run's checkpoint
  $R/venv/bin/torchrun --standalone --nproc_per_node $NG dynamics.py --name s150ev --size 150 --bs $(( 384 / NG )) \
    --init $R/init_s150.pt --ev_frac 0.125 --ev_w 4 >> $R/s150ev.log 2>&1; done
$PY hf.py backup; log "all done"
