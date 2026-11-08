#!/usr/bin/env bash
# MD-SOP v2.2：只有显式 --launch 才执行步骤5正式训练。
# 无参数只显示帮助；--preflight 只检查绑定/开发数据身份，不更新参数。
# 固定配对：GPU0跑w1，GPU1跑w15；A、B、C三组依次排队。
# 每组从已登记的未训练初始化开始，绝不加载v2.1已训练best/last。
# 中断后再次 --launch 恢复本轮last和数据队列；不可跨版本resume。
# GPU随机训练的下一步不承诺逐位一致：保留v2.1的cuDNN/dropout后端。
# 每轮loss/来源loss、验证RMSE、LR、样本暴露、PNG/SVG由训练器保存。
set -euo pipefail
PROJECT="/data/jiangbin/SOPPCL/MD-SOP"
TAG="v2_2_20261107"
CODE="${PROJECT}/code/${TAG}"
CONFIG="${PROJECT}/config/${TAG}/resolved_train_v2_2.yaml"
PYTHON="/data/jiangbin/DSSOP/envs/dssop-m0-py39-torch251-cu118-layered/bin/python"
LOGS="${PROJECT}/logs/${TAG}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
MODE="${1:---help}"
if [[ "${MODE}" == "--help" ]]; then
  echo "Usage: bash ${CODE}/run_matrix_v2_2.sh --preflight | --launch"
  echo "--preflight：不训练。--launch：执行六组正式训练（需步骤5授权）。"
  exit 0
fi
if [[ "$(id -un)" != "amax" ]]; then echo "Run as amax, not root." >&2; exit 2; fi
if [[ "${MODE}" != "--preflight" && "${MODE}" != "--launch" ]]; then echo "Unknown option" >&2; exit 2; fi
for EXP in A_MD_w1 A_MD_w15 B_SOPPE_w1 B_SOPPE_w15 C_MIX_w1 C_MIX_w15; do
  "${PYTHON}" "${CODE}/03_train_md_sop_v2_2.py" --config "${CONFIG}" --experiment-id "${EXP}" --device cuda:0 --validate-only
done
if [[ "${MODE}" == "--preflight" ]]; then exit 0; fi
# 这里只检查已有计算任务，不终止其它进程；若有任务，先由使用者协调GPU。
ACTIVE="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
if [[ -n "${ACTIVE}" ]]; then echo "GPU compute processes exist; no training launched: ${ACTIVE}" >&2; exit 3; fi
mkdir -p "${LOGS}"
exec > >(tee -a "${LOGS}/step5_matrix_v2_2_20261107.log") 2>&1
run_pair() {
  local e0="$1" e1="$2"
  "${PYTHON}" "${CODE}/03_train_md_sop_v2_2.py" --config "${CONFIG}" --experiment-id "${e0}" --device cuda:0 --resume auto >> "${LOGS}/${e0}_v2_2_20261107.log" 2>&1 &
  local p0=$!
  "${PYTHON}" "${CODE}/03_train_md_sop_v2_2.py" --config "${CONFIG}" --experiment-id "${e1}" --device cuda:1 --resume auto >> "${LOGS}/${e1}_v2_2_20261107.log" 2>&1 &
  local p1=$!
  set +e
  wait "${p0}"; local s0=$?
  wait "${p1}"; local s1=$?
  set -e
  if [[ "${s0}" -ne 0 || "${s1}" -ne 0 ]]; then echo "Pair failed: ${e0}=${s0}, ${e1}=${s1}"; return 1; fi
}
run_pair A_MD_w1 A_MD_w15
run_pair B_SOPPE_w1 B_SOPPE_w15
run_pair C_MIX_w1 C_MIX_w15
"${PYTHON}" "${CODE}/03b_summarize_training_v2_2.py" --config "${CONFIG}"
echo "Six development runs complete. No test evaluation was started."
