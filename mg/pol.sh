#!/bin/bash
# Policy-head pretraining on the dyna-rl node (job 6956433): s150w (best 150M world model) + a next-action head trained by
# behaviour cloning on the human videos, jointly with the world-model loss (dynamics.py --pol_w). Everything lives in
# node-local /tmp: data + s150w come from HF, checkpoints/dashboard go back to the HF bucket under runs_pol/ and dash_pol/.
#   ssh node 'setsid nohup bash /tmp/slither/mg/code/pol.sh > /tmp/slither/mg/pol.log 2>&1 &'
set -uo pipefail
R=/tmp/slither/mg; cd $R/code
export MG_ROOT=$R HF_HOME=$R/hf_home HF_TOKEN=$(cat $R/.hf_token) HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 UV_CACHE_DIR=$R/uvcache
export MG_HF_RUNS=runs_pol MG_HF_DASH=dash_pol MG_HF_EVERY=1800 MG_HF_SKIP=s150w MG_HF_RESTORE="data/*,runs_b/s150w/*,runs_pol/*,dash_pol/**"
source /etc/profile.d/modules.sh 2>/dev/null; module load uv/0.9.18
NG=$(nvidia-smi -L | wc -l); log() { echo "[$(date +%F' '%T)] $*"; }
log "node $(hostname), $NG GPUs, $(nproc) CPUs"
if [ ! -x $R/venv/bin/python ]; then
  log "venv"; uv venv -q $R/venv --python 3.11 && uv pip install -q -p $R/venv/bin/python "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 && uv pip install -q -p $R/venv/bin/python numpy pillow huggingface_hub && rm -rf $R/uvcache
fi
PY=$R/venv/bin/python
if [ ! -f $R/data/tokens.npy ]; then log "restore from HF"; $PY hf.py restore || exit 1
  mkdir -p $R/runs; [ -d $R/runs_b/s150w ] && mv $R/runs_b/s150w $R/runs/s150w
  for d in $R/runs_pol/*/; do [ -d "$d" ] && mv $d $R/runs/; done                 # this run's own earlier checkpoint wins
  [ -d $R/dash_pol ] && { mkdir -p $R/dash; cp -a $R/dash_pol/. $R/dash/; }; rm -rf $R/runs_b $R/runs_pol $R/dash_pol
fi
[ -f $R/runs/s150w/ckpt.pt ] || { log "s150w checkpoint missing"; exit 1; }
$PY hf.py loop > $R/backup.log 2>&1 &
log "s150p: s150w + policy head on $NG GPUs"
$R/venv/bin/torchrun --standalone --nproc_per_node $NG dynamics.py --name s150p --size 150 --bs ${BS:-64} \
  --init $R/runs/s150w/ckpt.pt --pol_w ${POL_W:-0.5} --lr ${LR:-1e-4} --epochs 3 --hours ${HOURS:-12} >> $R/s150p.log 2>&1
$PY hf.py backup; log "done"
