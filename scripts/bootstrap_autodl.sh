#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFP_DIR="${PROJECT_DIR}/SFP_ICCV"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

mkdir -p /root/autodl-tmp/src /root/autodl-tmp/data "${HF_HOME}"

git config --global http.version HTTP/1.1
git config --global --add safe.directory '*'

echo "[SFP-DeFT] project: ${PROJECT_DIR}"
echo "[SFP-DeFT] SFP source: ${SFP_DIR}"
echo "[SFP-DeFT] HF cache: ${HF_HOME}"

python -m pip install -U pip
python -m pip install -U huggingface_hub

if [ -f "${SFP_DIR}/environment.yml" ]; then
  echo "[SFP-DeFT] environment.yml exists at ${SFP_DIR}/environment.yml"
  echo "[SFP-DeFT] If the current Python env is empty, run:"
  echo "  conda env create -f ${SFP_DIR}/environment.yml"
  echo "  conda activate sfp_ovss"
fi

python -m py_compile \
  "${SFP_DIR}/SFP_segmentor.py" \
  "${SFP_DIR}/eval.py" \
  "${SFP_DIR}/resfp/config.py" \
  "${SFP_DIR}/resfp/density_tracing.py" \
  "${SFP_DIR}/resfp/cli.py"

cat <<EOF

[SFP-DeFT] bootstrap finished.

Before evaluation, edit dataset roots in:
  ${SFP_DIR}/configs/*.py

Example baseline:
  cd ${SFP_DIR}
  python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_sfp

Example DeFT:
  cd ${SFP_DIR}
  python eval.py --config ./configs/cfg_voc21.py --work-dir ./work_logs/voc21_deft --deft

Example matrix:
  cd ${PROJECT_DIR}
  python tools/run_eval_matrix.py \\
    --workdir ${SFP_DIR} \\
    --base-cmd "python eval.py --config ./configs/cfg_voc21.py" \\
    --work-root ${SFP_DIR}/work_logs/voc21_matrix

EOF
