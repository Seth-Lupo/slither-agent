#!/bin/bash
# Local control for the cluster run. Nothing lives on NFS: the job script goes in over stdin,
# code and the HF token are pushed straight into the node's /tmp once the job starts.
#   ./ctl.sh submit [gpus] [type] [node], wait for a node, push code (env: CPUS, MEM, NOPUSH=1)     ./ctl.sh push      re-push code
#   ./ctl.sh sync            dashboard sync + server on :8770        ./ctl.sh log       tail pipeline log
#   ./ctl.sh play [run]      live play tunnel on :8771               ./ctl.sh sh CMD    run CMD on the node
#   ./ctl.sh keep            chain every job (.spec*) past the 48 h limit: successor queued, code pushed when it starts
# JOB=ev ./ctl.sh ...       same commands for a second job; MAIN=big.sh ./ctl.sh submit ... runs big.sh (s400) as the job script
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); CFG=~/video-gen/.ssh_config; STATE=$HERE/.job${JOB:+.$JOB}; MAIN=${MAIN:-${JOB:+ab.sh}}; MAIN=${MAIN:-pipeline.sh}
ssh_login() { ssh -F $CFG vg-login "$@"; }
node() { cut -d' ' -f2 $STATE; }
on_node() { ssh -F $CFG -J vg-login slupo01@$(node) "$@"; }
SPEC=$HERE/.spec${JOB:+.$JOB}
sbatch_job() {  # gpus type node main cpus mem [after-job]
  ssh_login "sbatch --parsable ${7:+--dependency=afterany:$7}" <<EOF
#!/bin/bash
#SBATCH -J slither-mg -p gpu --qos=normal --gres=gpu:$2:$1 ${3:+-w $3} -c $5 --mem=$6 -t 2-00:00:00 -o /tmp/slither-mg-%j.out
mkdir -p /tmp/slither/mg/code
until [ -f /tmp/slither/mg/code/$4 ]; do sleep 5; done
exec bash /tmp/slither/mg/code/$4 >> /tmp/slither/mg/pipeline.log 2>&1
EOF
}

case ${1:-} in
submit)
  G=${2:-7}; TYPE=${3:-rtx_pro_6000}; NODE=${4:-}
  jid=$(sbatch_job $G $TYPE "$NODE" $MAIN ${CPUS:-$((G * 16))} ${MEM:-900G})
  echo "job $jid submitted"; echo "$MAIN $G $TYPE ${CPUS:-$((G * 16))} ${MEM:-900G}" > $SPEC
  until n=$(ssh_login "squeue -h -j $jid -o '%T %N'") && [[ $n == RUNNING* ]]; do echo "  ${n:-?}"; sleep 30; done
  echo "$jid ${n#RUNNING }" > $STATE; echo "running on $(node)"; [ -n "${NOPUSH:-}" ] || { sleep 5; "$0" push; } ;;
keep)   # Slurm caps jobs at 48 h: queue a successor behind each job (afterany), push code when it starts, repeat forever
  while true; do
    for S in $HERE/.spec $HERE/.spec.*; do [ -f $S ] || continue
      J=${S#$HERE/.spec}; J=${J#.}; ST=$HERE/.job${J:+.$J}; NX=$HERE/.next${J:+.$J}
      read -r M_ G_ T_ C_ R_ < $S; cur=$(cut -d' ' -f1 $ST)
      if [ ! -f $NX ]; then nj=$(sbatch_job $G_ $T_ "" $M_ $C_ $R_ $cur 2>/dev/null) && [[ $nj =~ ^[0-9]+$ ]] && { echo $nj > $NX; echo "$(date +%T) ${J:-main}: successor $nj queued after $cur"; }; continue; fi
      nj=$(cat $NX); out=$(ssh_login "squeue -h -j $nj -o '%T %N' 2>/dev/null; echo END" 2>/dev/null) || continue
      [[ $out == *END* ]] || continue; st=${out%END}; st=$(echo $st)
      if [[ $st == RUNNING* ]]; then echo "$nj ${st#RUNNING }" > $ST; rm -f $NX; echo "$(date +%T) ${J:-main}: $nj running on ${st#RUNNING }"
        sleep 20; JOB=$J "$0" push || echo "$(date +%T) ${J:-main}: push failed"
      elif [ -z "$st" ]; then rm -f $NX; echo "$(date +%T) ${J:-main}: successor $nj vanished, requeueing"; fi
    done
    sleep 120
  done ;;
push)
  on_node "mkdir -p /tmp/slither/mg/code"
  rsync -a -e "ssh -F $CFG -J vg-login" $HERE/*.py $HERE/*.sh slupo01@$(node):/tmp/slither/mg/code/
  (set -a; source ~/slither/.env; printf '%s\n' "$HF_TOKEN") | on_node 'umask 077; cat > /tmp/slither/mg/.hf_token'
  echo "code pushed to $(node)" ;;
sync)
  (cd $HERE/dash && python3 -m http.server 8770 --bind 127.0.0.1 >/dev/null 2>&1 &)
  echo "dashboard: http://localhost:8770"
  E="ssh -F $CFG -J vg-login -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
  while true; do                    # main job -> dash/; side job s150 (s150.sh) -> its runs under dash/dyn/, status merged in
    X=(); [ -f $HERE/.job.s150 ] && X=(--exclude dyn/s150/ --exclude dyn/s150x/)
    timeout 300 rsync -a --exclude index.html ${X[@]+"${X[@]}"} -e "$E" slupo01@$(node):/tmp/slither/mg/dash/ $HERE/dash/ 2>/dev/null || echo "$(date +%T) sync failed"
    if [ -f $HERE/.job.s150 ]; then n2=$(cut -d' ' -f2 $HERE/.job.s150)
      { timeout 300 rsync -a --exclude tok/ -e "$E" slupo01@$n2:/tmp/slither/mg/dash/dyn/ $HERE/dash/dyn/ && timeout 60 rsync -a -e "$E" slupo01@$n2:/tmp/slither/mg/dash/status.json $HERE/dash/status_b.json; } 2>/dev/null || echo "$(date +%T) sync s150 failed"
      python3 -c "import json;a=json.load(open('$HERE/dash/status.json'));b=json.load(open('$HERE/dash/status_b.json'));a.update({k:v for k,v in b.items() if k.startswith('dyn_s150')});json.dump(a,open('$HERE/dash/status.json.tmp','w'))" 2>/dev/null && mv $HERE/dash/status.json.tmp $HERE/dash/status.json
    fi
    sleep 60
  done ;;
log) on_node "tail -n ${2:-40} /tmp/slither/mg/pipeline.log; ls /tmp/slither/mg/*.log 2>/dev/null | xargs -r tail -n 3" ;;
play)
  on_node "cd /tmp/slither/mg/code && MG_ROOT=/tmp/slither/mg nohup ../venv/bin/python play.py --run ${2:-s150} > ../play.log 2>&1 &"
  echo "http://localhost:8771  (Ctrl-C closes the tunnel)"; ssh -F $CFG -J vg-login -N -L 8771:127.0.0.1:8771 slupo01@$(node) ;;
sh) shift; on_node "$@" ;;
*) sed -n 2,7p "$0" ;;
esac
