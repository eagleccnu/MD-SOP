#!/usr/bin/env bash
# 步骤5启动入口。默认无副作用；只有 --launch 才新建后台tmux会话。
# 查看：tmux attach -t mdsop_v2_2_20261107
# 离开但保留任务：Ctrl-b，然后按d。不要使用Ctrl-c终止训练。
set -euo pipefail
SESSION="mdsop_v2_2_20261107"
SCRIPT="/data/jiangbin/SOPPCL/MD-SOP/code/v2_2_20261107/run_matrix_v2_2.sh"
if [[ "${1:---help}" != "--launch" ]]; then
  echo "步骤4仅生成此脚本，未启动训练。步骤5使用：bash $0 --launch"
  exit 0
fi
if [[ "$(id -un)" != "amax" ]]; then echo "Run as amax." >&2; exit 2; fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then echo "Session exists: ${SESSION}; inspect it instead of overwriting." >&2; exit 2; fi
tmux new-session -d -s "${SESSION}" "bash ${SCRIPT} --launch; code=\$?; echo MatrixExit=\${code}; exec bash"
echo "Started ${SESSION}. Attach with: tmux attach -t ${SESSION}"
