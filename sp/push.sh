#!/bin/bash
# Local: copy the self-play code to the GPU node (/tmp/slither/sp/code) through Slurm.
CFG=~/video-gen/.ssh_config; J=${JOB:-6956433}
COPYFILE_DISABLE=1 tar cf - -C ~/slither/sp --exclude __pycache__ --no-xattrs . | ssh -F $CFG -o ConnectTimeout=20 vg-login timeout 100 srun --jobid=$J --overlap -N1 -n1 bash -c "'mkdir -p /tmp/slither/sp/code && tar xf - -C /tmp/slither/sp/code'" 2>&1 | grep -v '^\*\*'
echo pushed
