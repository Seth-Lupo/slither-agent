#!/bin/bash
# Set up the harness on a cluster node, everything under node-local /tmp (NFS home is tiny; /tmp is wiped with the job).
#   bash setup_node.sh            then: /tmp/slither/harness/.venv/bin/python /tmp/slither/harness/scripts/run.py --sessions 8 --seconds 120
set -euo pipefail
R=/tmp/slither/harness; mkdir -p $R
export PLAYWRIGHT_BROWSERS_PATH=$R/browsers UV_CACHE_DIR=$R/uvcache
source /etc/profile.d/modules.sh 2>/dev/null; module load uv/0.9.18 2>/dev/null || true
cd $R
[ -x .venv/bin/python ] || uv venv -q .venv --python 3.11
uv pip install -q -p .venv/bin/python -r requirements.txt
.venv/bin/playwright install chromium-headless-shell 2>&1 | tail -1
# system libraries Chromium needs; if this lists any "not found", ask research computing or use the apptainer image
ldd $R/browsers/chromium_headless_shell-*/chrome-*/chrome-headless-shell 2>/dev/null | grep "not found" || echo "chromium libs ok"
curl -s -o /dev/null -w "slither.io reachable: HTTP %{http_code}\n" --max-time 10 http://slither.io/ || echo "slither.io NOT reachable from this node"
echo "export PLAYWRIGHT_BROWSERS_PATH=$R/browsers"
