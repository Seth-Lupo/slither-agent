#!/bin/bash
# Fallback while HF uploads are blocked: every 2 h copy each job's checkpoints to ckpt_backup/ on this Mac (overwritten in place).
HERE=$(cd "$(dirname "$0")" && pwd); E="ssh -F $HOME/video-gen/.ssh_config -J vg-login -o ConnectTimeout=20 -o ServerAliveInterval=15"
while true; do
  for J in "" .s150; do [ -f $HERE/.job$J ] || continue; n=$(cut -d' ' -f2 $HERE/.job$J); d=$HERE/ckpt_backup/${J:-.s400}; d=${d/\/./\/}node; mkdir -p $d
    if [ -z "$J" ]; then inc=(--include='s400/***' --include='s400x/***'); else inc=(--include='s150/***' --include='s150x/***'); fi
    timeout 3600 rsync -a "${inc[@]}" --exclude='*/*.tmp' --exclude='*' -e "$E" slupo01@$n:/tmp/slither/mg/runs/ $d/ 2>/dev/null \
      && echo "$(date +%T) mirrored $n -> $d" || echo "$(date +%T) mirror $n failed"
  done
  sleep 7200
done
