#!/bin/bash
# Second job (RTX PRO node): finish s150's 4-epoch schedule from its HF checkpoint, then keep going as s150w: fine-tuned from
# s150x (s150 continued at its final constant LR, plateaued at val 2.6285) with each token's loss weighted by 1 + its
# unexplained energy (dynamics.py energy(): what motion can't explain, e.g. a worm disintegrating), same constant LR.
# Never touches the s400 job. s150w never finishes: `ctl.sh keep` queues a successor job, which resumes it.
# Checkpoints and dashboard go to runs_b/ and dash_b/ on HF, so the s400 job's stale s150 copy can't overwrite them.
set -uo pipefail
R=/tmp/slither/mg; cd $R/code
export MG_ROOT=$R HF_HOME=$R/hf_home HF_TOKEN=$(cat $R/.hf_token) HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 UV_CACHE_DIR=$R/uvcache
export MG_HF_RUNS=runs_b MG_HF_DASH=dash_b MG_HF_EVERY=1800 MG_HF_RESTORE="data/*,runs/s150/*,dash/dyn/s150/*,runs_b/*,dash_b/**"
source /etc/profile.d/modules.sh 2>/dev/null; module load uv/0.9.18
NG=$(nvidia-smi -L | wc -l); BS=$(( 384 / NG )); log() { echo "[$(date +%F' '%T)] $*"; }
log "node $(hostname), $NG GPUs x bs $BS, $(nproc) CPUs"
if [ ! -x $R/venv/bin/python ]; then
  log "venv"; uv venv -q $R/venv --python 3.11 && uv pip install -q -p $R/venv/bin/python "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 && uv pip install -q -p $R/venv/bin/python numpy pillow huggingface_hub && rm -rf $R/uvcache
fi
PY=$R/venv/bin/python
if [ ! -f $R/data/tokens.npy ]; then log "restore"; $PY hf.py restore || exit 1   # this job's own backups (runs_b, dash_b) win
  for d in $R/runs_b/*/; do [ -d "$d" ] && { n=$(basename $d); rm -rf $R/runs/$n; mkdir -p $R/runs; mv $d $R/runs/$n; }; done
  [ -d $R/dash_b ] && { mkdir -p $R/dash; cp -a $R/dash_b/. $R/dash/; }; rm -rf $R/runs_b $R/dash_b
fi
$PY hf.py loop > $R/backup.log 2>&1 &
for i in 1 2 3; do [ -f $R/runs/s150/DONE ] && break; log "dynamics ($i): s150 resume on $NG GPUs"
  $R/venv/bin/torchrun --standalone --nproc_per_node $NG dynamics.py --name s150 --size 150 --bs $BS >> $R/s150.log 2>&1; done
$PY hf.py backup
for i in 1 2 3; do [ -f $R/runs/s150w/DONE ] && break; log "dynamics ($i): s150w (energy-weighted loss) on $NG GPUs"
  $R/venv/bin/torchrun --standalone --nproc_per_node $NG dynamics.py --name s150w --size 150 --bs $BS \
    --init $R/runs/s150x/ckpt.pt --epochs 1000 --flat 0.1 --energy 1 >> $R/s150w.log 2>&1; done
$PY hf.py backup; log "all done"
