#!/bin/bash
#SBATCH --job-name=graphlex-gnn
#SBATCH --account=coml-dhs
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/coml-dhs/stat0278/graphlex/logs/gnn_%A_%a.out
#SBATCH --error=/data/coml-dhs/stat0278/graphlex/logs/gnn_%A_%a.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=flaxman@gmail.com

# Few-shot trained GNN baselines (GIN + GCN) at the graphlex sweep's matched
# splits. GIN/GCN on TUDatasets is LIGHT; one L40S is plenty (CPU would also
# work). Idempotent/resumable: gnn_baseline.py SKIPs any dataset whose result
# JSON already exists, so re-submitting only fills gaps.
#
# --- SMOKE (1 dataset per submit, 1 seed, few epochs) ---
#   NOTE: sbatch --export splits on commas, so a multi-dataset DATASETS list in
#   --export gets mangled (only the first survives). For a multi-dataset smoke,
#   either submit one dataset per sbatch, or use the array mode below. gpu:0 runs
#   it CPU-only (GIN/GCN is light) so it schedules even when L40S nodes are busy.
#   ssh htc 'cd /data/coml-dhs/stat0278/graphlex && \
#     sbatch --gres=gpu:0 --time=00:20:00 \
#     --export=ALL,DATASETS=MUTAG,SEEDS=11,EPOCHS=40 eval/gnn_baseline.slurm'
#
# --- FULL SWEEP as a job array over the 30 datasets (one dataset per task) ---
#   ssh htc 'cd /data/coml-dhs/stat0278/graphlex && \
#     sbatch --array=0-29 eval/gnn_baseline.slurm'
#
# --- FULL SWEEP as a single job (loops over all manifest datasets) ---
#   ssh htc 'cd /data/coml-dhs/stat0278/graphlex && \
#     sbatch --time=04:00:00 eval/gnn_baseline.slurm'
#
# Env overrides (via --export=ALL,KEY=val): DATASETS, SEEDS, EPOCHS,
# FULL_SUPERVISION=1 (also run the 80/10/10 upper-bound GIN reference), FORCE=1.

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 uv/0.2.30-GCCcore-13.3.0

export UV_CACHE_DIR=/data/coml-dhs/stat0278/.uv-cache
export TMPDIR=/data/coml-dhs/stat0278/.tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

DATA=/data/coml-dhs/stat0278/graphlex
export TU_ROOT="$DATA/tudata"
export GNN_OUT="$DATA/results/gnn"
export SWEEP_MANIFEST="$DATA/results/sweep_manifest.json"
mkdir -p "$GNN_OUT" "$DATA/logs"

: "${SEEDS:=11,22,33}"
: "${EPOCHS:=200}"
EXTRA=""
[ "${FULL_SUPERVISION:-0}" = "1" ] && EXTRA="$EXTRA --full-supervision"
[ "${FORCE:-0}" = "1" ] && EXTRA="$EXTRA --force"

# Resolve the dataset list. Priority:
#   1. DATASETS env (explicit comma list)  -> used as-is
#   2. SLURM_ARRAY_TASK_ID set             -> the Nth dataset from the manifest
#   3. neither                             -> all manifest datasets (single job loop)
ALL_DATASETS="$(python - <<'PY'
import json, os
mf = os.environ["SWEEP_MANIFEST"]
print(",".join(sorted(json.load(open(mf))["baselines"].keys())))
PY
)"

if [ -n "${DATASETS:-}" ]; then
  RUN_DS="$DATASETS"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  RUN_DS="$(python - "$ALL_DATASETS" "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
ds = sys.argv[1].split(",")
i = int(sys.argv[2])
if i >= len(ds):
    print("")
else:
    print(ds[i])
PY
)"
  if [ -z "$RUN_DS" ]; then
    echo "array index ${SLURM_ARRAY_TASK_ID} out of range (have ${ALL_DATASETS}); nothing to do"
    exit 0
  fi
else
  RUN_DS="$ALL_DATASETS"
fi

echo "=== graphlex GNN baseline ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_TASK_ID:-none}"
echo "datasets=$RUN_DS"
echo "seeds=$SEEDS epochs<=$EPOCHS extra=$EXTRA"
echo "TU_ROOT=$TU_ROOT  GNN_OUT=$GNN_OUT"
nvidia-smi || true
echo "============================="

python -u eval/gnn_baseline.py \
  --datasets "$RUN_DS" \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS" \
  --tu-root "$TU_ROOT" \
  --out "$GNN_OUT" \
  --manifest "$SWEEP_MANIFEST" \
  $EXTRA

echo "SLURM JOB DONE"
