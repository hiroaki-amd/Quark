#!/bin/bash
# Generate RULER (NVIDIA/RULER, https://github.com/NVIDIA/RULER) validation data for all 13
# standard tasks, at a given context length, tokenized for a given model.
#
# Requires an NVIDIA/RULER checkout (its scripts/data/prepare.py is what actually generates
# the data): https://github.com/NVIDIA/RULER
#
# prepare.py internally shells out to `python <generator>.py ...` per task (bare "python", not
# sys.executable, and *not* whatever RULER_PYTHON invoked prepare.py itself) -- so whatever
# environment has RULER's own deps (pyyaml, numpy, tenacity, wonderwords, etc. -- not
# necessarily the same environment calibrate_model.py needs) must have its bin directory first
# on PATH, not just be pointed at via RULER_PYTHON, or that nested subprocess resolves to a
# python lacking those deps.
#
# Usage: RULER_DATA_DIR=/path/to/RULER/scripts/data [RULER_PYTHON=/path/to/python] \
#   bash generate_ruler_data.sh <model> <context_length> <out_dir> [samples_per_task]
set -euo pipefail

MODEL=${1:?"Usage: $0 <model> <context_length> <out_dir> [samples_per_task]"}
CTX=${2:?"Usage: $0 <model> <context_length> <out_dir> [samples_per_task]"}
OUT_DIR=${3:?"Usage: $0 <model> <context_length> <out_dir> [samples_per_task]"}
N=${4:-64}
RULER_DATA_DIR=${RULER_DATA_DIR:?"Set RULER_DATA_DIR to your NVIDIA/RULER checkout's scripts/data directory"}
RULER_PYTHON=${RULER_PYTHON:-python}
export PATH="$(dirname "$(command -v "${RULER_PYTHON}")"):${PATH}"

TASKS=(
    niah_single_1 niah_single_2 niah_single_3
    niah_multikey_1 niah_multikey_2 niah_multikey_3
    niah_multivalue niah_multiquery
    vt cwe fwe qa_1 qa_2
)

mkdir -p "${OUT_DIR}"
echo "Generating RULER data: model=${MODEL} ctx=${CTX} n=${N} out=${OUT_DIR}"

for TASK in "${TASKS[@]}"; do
    JSONL="${OUT_DIR}/${TASK}/validation.jsonl"
    if [ -f "${JSONL}" ] && [ "$(wc -l < "${JSONL}")" -ge "${N}" ]; then
        echo "[skip] ${TASK} already has $(wc -l < "${JSONL}") samples"
        continue
    fi
    echo "[gen] ${TASK}"
    (cd "${RULER_DATA_DIR}" && "${RULER_PYTHON}" prepare.py \
        --save_dir "${OUT_DIR}" \
        --benchmark synthetic \
        --task "${TASK}" \
        --tokenizer_path "${MODEL}" \
        --tokenizer_type hf \
        --max_seq_length "${CTX}" \
        --num_samples "${N}" \
        --model_template_type base) 2>&1 | tail -5
done

echo "Done. Sample counts:"
for TASK in "${TASKS[@]}"; do
    JSONL="${OUT_DIR}/${TASK}/validation.jsonl"
    [ -f "${JSONL}" ] && echo "  ${TASK}: $(wc -l < "${JSONL}")" || echo "  ${TASK}: MISSING"
done
