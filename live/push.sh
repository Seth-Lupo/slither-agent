#!/bin/bash
# Local: copy the live code to the GPU node (/tmp/slither/live/code) through Slurm.
CFG=~/video-gen/.ssh_config; J=${JOB:-6956433}
COPYFILE_DISABLE=1 tar cf - -C ~/slither/live --exclude __pycache__ --no-xattrs . | ssh -F $CFG -o ConnectTimeout=20 vg-login timeout 100 srun --jobid=$J --overlap -N1 -n1 bash -c "'mkdir -p /tmp/slither/live/code && tar xf - -C /tmp/slither/live/code'" 2>&1 | grep -v '^\*\*'
echo pushed
