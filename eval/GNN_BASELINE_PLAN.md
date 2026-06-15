# GNN baseline (GIN + GCN) — lab note

Adds **trained** GNN arms to the graphlex sweep. The existing sweep has only
logreg-on-`facts()` and majority — no trained GNN. This fills that gap with
few-shot GIN + GCN trained on the *exact same* labeled graphs/class and scored
on the *exact same* query graphs with the *exact same* balanced-accuracy metric,
so the new rows are apples-to-apples with the LLM / logreg / majority arms.

This is a lab note, not paper prose.

## Files

- `eval/gnn_baseline.py` — trainer. Few-shot GIN+GCN at matched splits;
  writes one results JSON per dataset. Optional `--full-supervision` 80/10/10
  GIN reference row (clearly a *different* setting).
- `eval/gnn_baseline.slurm` — the job (smoke + array + single-job modes).
- `eval/gnn_baseline_install.slurm` — one-time uv venv build.

## HTC / cluster workflow we settled on

- **SSH alias:** `htc` (works from clpc95/clpc35). `arc` (CPU) also works; GIN/GCN
  is light enough that CPU is fine, but the job defaults to one L40S on `short`.
- **NFS home is NOT visible from htc** (`/users/setman/...` absent on the login
  node). So we do **not** rely on the NFS-home mirror — code + data are pushed to
  `/data` by `rsync` from clpc95 directly over ssh.
- **Project dir (new):** `/data/coml-dhs/stat0278/graphlex/` — reuses stat0278's
  `coml-dhs` 5 TB scratch (4.9 TB free), in a fresh subdir separate from
  `learning-to-explain`. Layout:
  ```
  graphlex/
  ├── .venv/                 # uv venv (torch 2.7.1+cu126 + torch_geometric)
  ├── eval/gnn_baseline*.py/.slurm
  ├── graphlex/              # the package (for eval/_common imports)
  ├── tudata/                # TUDataset cache (rsynced from clpc95:/home/scratch/tudata)
  ├── results/
  │   ├── sweep_manifest.json   # copy of the sweep manifest (split parity check + ds list)
  │   └── gnn/<DATASET>.json    # per-dataset GNN results
  └── logs/                  # slurm stdout/stderr
  ```
- **Account:** `coml-dhs` (the scratch group's account; `coml-fado` is the other
  option — the foundation sprint uses it). Override per submit with
  `--account=coml-fado` if desired.
- **Submit form (always `cd` first so `$SLURM_SUBMIT_DIR` is the repo):**
  ```
  ssh htc 'cd /data/coml-dhs/stat0278/graphlex && sbatch [overrides] eval/gnn_baseline.slurm'
  ```

### Sync commands (clpc95 → htc)

```
# code:
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
  --exclude 'logs' --exclude '.pytest_cache' --exclude '*.egg-info' \
  /home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex/ \
  htc:/data/coml-dhs/stat0278/graphlex/

# datasets (35 G; DBLP_v1 alone is 32 G and dominates the transfer time):
rsync -az /home/scratch/tudata/ htc:/data/coml-dhs/stat0278/graphlex/tudata/

# the sweep manifest (needed for split-parity assertion + dataset enumeration):
rsync -az /home/scratch/bench_out/sweep/manifest.json \
  htc:/data/coml-dhs/stat0278/graphlex/results/sweep_manifest.json
```

## Env recipe (verified)

modules + uv, mirroring l2e's verified stack:

```
module load Python/3.12.3-GCCcore-13.3.0 uv/0.2.30-GCCcore-13.3.0
export UV_CACHE_DIR=/data/coml-dhs/stat0278/.uv-cache
export TMPDIR=/data/coml-dhs/stat0278/.tmp
export UV_HTTP_TIMEOUT=600     # nvidia-cudnn-cu12 (~0.5 G) times out at the 30s default
uv venv --python "$(which python3)" .venv && source .venv/bin/activate
uv pip install "torch==2.7.1+cu126" --index-url https://download.pytorch.org/whl/cu126
uv pip install "numpy<2.5" scikit-learn networkx torch_geometric
# optional compiled exts (NOT required — PyG>=2.5 has native scatter):
uv pip install pyg-lib torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.7.1+cu126.html || true
uv pip install -e .   # graphlex package
```

Gotchas hit: (1) plain `torch==2.7.1` won't resolve on the cu126 wheel index —
it only carries the `+cu126` local-version tag, so request `torch==2.7.1+cu126`.
(2) default `UV_HTTP_TIMEOUT=30` is too short for the bundled CUDA wheels.
Install was moved off `devel` (10-min cap) onto `short` (30-min `--time`) to give
the first cold download room.

## Featurization (consistent, documented)

Decided **once per dataset** from a sample of graphs (`dataset_uses_node_cats`):

- **Clean one-hot categorical node features present** (the same condition
  `eval/_common.node_cats` uses — rows sum to 1, values in {0,1}): keep the
  one-hot node labels. This gives the GNN the **same** node-type information the
  classical/LLM arms get via `comp()` / the `'type'` attribute. Used by the
  chemistry/bio datasets (MUTAG, NCI1, PROTEINS, ENZYMES, …).
- **No usable node features** (IMDB, COLLAB, the ego-net / social datasets): a
  **degree one-hot** feature (degree clipped to 50, one-hot to width 51) — the
  standard TU fallback (GIN paper appendix; PyG benchmark scripts use degree /
  constant features for featureless TU graphs).

All graphs within a dataset use the single dataset-wide scheme, so `in_dim` is
constant.

## Architecture + hyperparameters (FIXED across all 30 datasets, no tuning)

| | value |
|---|---|
| GIN | `N_LAYERS=4` × `GINConv(MLP[d→64→64], train_eps=True)` + BatchNorm + ReLU, **global add pool**, MLP head (64→64→C) with dropout 0.5 |
| GCN | `N_LAYERS=4` × `GCNConv(·→64)` + BatchNorm + ReLU, **global mean pool**, same head |
| hidden | 64 |
| optimizer | Adam, lr 1e-2, weight decay 5e-4 |
| epochs | ≤ 200, **early stop** patience 30 on a held-out shot slice |
| val split | 25% of the shots, stratified per class (for early stopping only) |
| class weights | inverse-frequency CE (counters shot imbalance on many-class sets) |
| batch | 32 |

No per-dataset tuning — a single config runs everywhere. This is deliberate: the
few-shot GNN is **expected to struggle** at K≈5 shots/class (GNNs are data
hungry). That weak result is the legitimate, honest label-efficiency point, not a
bug — do **not** tune it into looking good.

## Matched-split protocol (provably identical to the LLM/logreg arms)

`gnn_baseline.py` reproduces `sweep.py` / `balanced_rescore.py`'s split logic
byte-for-byte:

```
cap  = min(len(ds), POOL_CAP=4000)
idx  = [i<cap : num_nodes>=3 and n_edges>=1]
spc  = max(2, min(SHOTS_PER_CLASS=5, MAX_SHOTS=60 // n_classes))
per seed in [11,22,33]:
  rng = RandomState(seed)
  pos[c] = rng.permutation(positions with y==c)
  shot += pos[c][:spc]; q += pos[c][spc:]
  rng.shuffle(q); q = q[:NQ=40]; rng.shuffle(shot)
```

Shots → GNN training (with the 25% early-stop slice carved out of them); `q` →
queries. **Parity assertion:** for each (dataset, seed) we rebuild the query
truth labels and assert they equal the manifest's stored `truth` list; a mismatch
aborts with `SPLIT MISMATCH`. So the GNN provably sees the same graphs as the
LLM / logreg arms. Balanced accuracy uses `_common.bal_acc` — the identical
metric.

## How results fold into the figures

Each `results/gnn/<DATASET>.json` holds
`mean.{gin,gcn} = [mean_balacc, std]` over seeds plus per-seed values. To add
GIN/GCN as heatmap arms in `eval/make_figures.py`, extend its `METHODS` list with
`("gin","GIN (trained)")` and `("gcn","GCN (trained)")` and have
`compute_sweep_table` read `mean.gin[0]` / `mean.gcn[0]` from these JSONs (keyed
by dataset). The regret panel then shows the trained-GNN arm directly against
the best non-LLM baseline and the LLM arms, which is the label-efficiency
headline: at 5 shots/class the trained GNN typically sits at/near chance while
logreg-on-`facts()` and the LLM do meaningfully better. (The figure wiring is
left as a one-liner for whoever regenerates figures, to avoid touching the
committed figure cache here.)

## Smoke run

`--export=ALL,DATASETS=MUTAG,IMDB-BINARY,SEEDS=11,EPOCHS=40` on `short`. See the
PR / commit message for the exact balanced-accuracy numbers from the clean smoke
run.

## Full sweep submit command

Array (one dataset per task, idempotent — skips datasets whose JSON exists):

```
ssh htc 'cd /data/coml-dhs/stat0278/graphlex && sbatch --array=0-29 eval/gnn_baseline.slurm'
```

or a single job that loops all 30 (bump `--time` since DBLP_v1 is large):

```
ssh htc 'cd /data/coml-dhs/stat0278/graphlex && sbatch --time=04:00:00 eval/gnn_baseline.slurm'
```

Add `--export=ALL,FULL_SUPERVISION=1` to also emit the 80/10/10 upper-bound GIN
reference row.
```
