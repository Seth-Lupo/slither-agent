#!/bin/bash
# Runs on the compute node inside the job. Each stage is skipped when its output already exists,
# so a resubmitted job (after `hf.py restore`) picks up where the last one stopped.
set -uo pipefail
R=/tmp/slither/mg; cd $R/code
export MG_ROOT=$R HF_TOKEN=$(cat $R/.hf_token) HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 UV_CACHE_DIR=$R/uvcache
source /etc/profile.d/modules.sh 2>/dev/null; module load uv/0.9.18 ffmpeg/7.0.2
NG=$(nvidia-smi -L | wc -l); log() { echo "[$(date +%F' '%T)] $*"; }
log "node $(hostname), $NG GPUs, $(nproc) CPUs"

if [ ! -x $R/venv/bin/python ]; then
  log "venv"; uv venv -q $R/venv --python 3.11 && uv pip install -q -p $R/venv/bin/python "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 && uv pip install -q -p $R/venv/bin/python numpy pillow huggingface_hub && rm -rf $R/uvcache
fi
PY=$R/venv/bin/python; TR="$R/venv/bin/torchrun --standalone"
$PY hf.py restore

if [ ! -f $R/data/tokens.npy ]; then
  [ -f $R/data/meta.json ] || { log "fetch"; $PY hf.py fetch && log "prep" && $PY prep.py; } || exit 1
  [ -f $R/data/tokenizer.done ] || { log "tokenizer"; $TR --nproc_per_node $NG tokenizer.py --hours 3 && touch $R/data/tokenizer.done; } || exit 1
  log "encode"; $TR --nproc_per_node $NG encode.py || exit 1
  rm -rf $R/src $R/data/tok $R/data/tmpfr $R/data/clips        # frees ~175 GB of RAM
  $PY hf.py backup
fi
$PY hf.py loop > $R/backup.log 2>&1 &

# Dynamics (s150 keeps a 384 global batch whatever the GPU count, so resumes stay on schedule): pilot (50M on 50 h) then 50M on all data on a quarter of the GPUs; the 150M main run on the rest.
P=$(( NG / 4 > 0 ? NG / 4 : 1 )); M=$(( NG - P ))
mkdir -p $R/runs
log "dynamics: s150 on $M GPUs, pilot50 -> s50 on $P"
( [ -f $R/runs/s150/DONE ] || CUDA_VISIBLE_DEVICES=$(seq -s, $P $((NG-1))) $TR --nproc_per_node $M dynamics.py --name s150 --size 150 --bs $(( 384 / M )) > $R/s150.log 2>&1 ) &
( G=$(seq -s, 0 $((P-1)))
  [ -f $R/runs/pilot50/DONE ] || CUDA_VISIBLE_DEVICES=$G $TR --nproc_per_node $P dynamics.py --name pilot50 --size 50 --data_hours 50 > $R/pilot50.log 2>&1
  [ -f $R/runs/s50/DONE ] || CUDA_VISIBLE_DEVICES=$G $TR --nproc_per_node $P dynamics.py --name s50 --size 50 > $R/s50.log 2>&1 ) &
wait
$PY hf.py backup; log "all done"
