#!/bin/bash
#SBATCH --job-name=gnn-install
#SBATCH --account=coml-dhs
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/coml-dhs/stat0278/graphlex/logs/install_%j.out
#SBATCH --error=/data/coml-dhs/stat0278/graphlex/logs/install_%j.err

# One-time uv venv build for the graphlex GNN baselines.
#   torch + torch_geometric (+ scatter/sparse via PyG's pyg-lib wheels),
#   numpy<2.5, scikit-learn, networkx, and graphlex itself (editable).
#
#   ssh htc 'cd /data/coml-dhs/stat0278/graphlex && sbatch eval/gnn_baseline_install.slurm'
#
# torch 2.7.1+cu126 matches the l2e verified stack (works on L40S/A100/H100;
# also CPU-only on arc). torch_geometric>=2.5 has native scatter, so for a
# CPU/GPU GIN/GCN we do NOT need the compiled torch_scatter/torch_sparse
# extensions -- PyG falls back to its built-in scatter. We install pyg-lib /
# torch_scatter / torch_sparse from the matching wheel index opportunistically
# (|| true) so the job still succeeds if those wheels are unavailable.

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0 uv/0.2.30-GCCcore-13.3.0

export UV_CACHE_DIR=/data/coml-dhs/stat0278/.uv-cache
export TMPDIR=/data/coml-dhs/stat0278/.tmp
# big CUDA dep wheels (nvidia-cudnn-cu12 ~0.5GB, cublas, etc.) blow the default
# 30s uv download timeout on the cluster network -> bump it generously.
export UV_HTTP_TIMEOUT=600
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

cd "$SLURM_SUBMIT_DIR"

if [ ! -d .venv ]; then
  uv venv --python "$(which python3)" .venv
fi
source .venv/bin/activate

# core stack (torch first, pinned to the l2e-verified cu126 build). The cu126
# wheel index only carries the +cu126 local-version tag, so we must request it
# explicitly (plain "torch==2.7.1" won't resolve against that index).
uv pip install "torch==2.7.1+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install "numpy<2.5" scikit-learn networkx
uv pip install torch_geometric

# optional compiled extensions (not required by GINConv/GCNConv on PyG>=2.5)
uv pip install pyg-lib torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.7.1+cu126.html || \
  echo "WARN: pyg-lib/scatter/sparse wheels unavailable; PyG native scatter will be used"

# graphlex itself (for _common.py's node_cats etc. -- code lives in this tree)
uv pip install -e . || echo "WARN: editable graphlex install failed (eval/_common is import-path based anyway)"

python - <<'PY'
import torch, torch_geometric, numpy, sklearn, networkx
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
print("numpy", numpy.__version__, "sklearn", sklearn.__version__, "nx", networkx.__version__)
from torch_geometric.nn import GINConv, GCNConv, global_add_pool
print("PyG GIN/GCN imports OK")
PY
echo "INSTALL DONE"
