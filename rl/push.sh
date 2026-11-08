#!/bin/bash
# Local: copy the RL code (and harness) to the GPU node through Slurm (direct ssh to the node can hang).
CFG=~/video-gen/.ssh_config; J=${JOB:-6956433}
S="ssh -F $CFG -o ConnectTimeout=20 vg-login timeout 100 srun --jobid=$J --overlap -N1 -n1"
COPYFILE_DISABLE=1 tar cf - -C ~/slither/rl --exclude dash/data --exclude 'dash/.sync*' --exclude __pycache__ --exclude "*.log" --no-xattrs . | $S bash -c "'mkdir -p /tmp/slither/rl2/code && tar xf - -C /tmp/slither/rl2/code'" 2>&1 | grep -v '^\*\*'
COPYFILE_DISABLE=1 tar cf - -C ~/slither/harness --exclude .venv --exclude __pycache__ --exclude 'episodes*' . | $S bash -c "'tar xf - -C /tmp/slither/harness'" 2>&1 | grep -v '^\*\*'
echo pushed
