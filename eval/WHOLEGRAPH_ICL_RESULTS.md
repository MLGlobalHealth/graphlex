# Whole-graph native-ICL FM (comparison #2): GILT — validated & swept (2026-06-15)

Comparison #2 for graphlex is a **pretrained, tuning-free, native-in-context
whole-graph foundation model** — the closest non-LLM analogue to graphlex+LLM
(both do few-shot whole-graph classification with **no per-task gradient step**).
The candidate is **GILT** (arXiv 2510.04567, "GILT: An LLM-Free, Tuning-Free Graph
Foundational Model for In-Context Learning"; code github.com/yiming421/inductnode;
checkpoint huggingface.co/fdsajkshf/gilt-checkpoint/gilt_model.pt). Given K labeled
SUPPORT graphs/class + query graphs, GILT predicts every query in ONE no-gradient
forward pass via a PFN transformer over per-graph prototype embeddings.

Harness (already written, run verbatim — not modified):
`eval/gilt_smoke_molhiv.py` (the gate) and `eval/gilt_icl.py` (the matched sweep).

## GILT smoke verdict — **PASS** (the checkpoint is REAL)

> My earlier lit review flagged the HF org (`fdsajkshf/gilt-checkpoint`) as
> placeholder-looking. The smoke test is the definitive check. It PASSES.

5-shot ogbg-molhiv, GILT's own featurization (raw 9-dim OGB atom feats → PCA128 →
L2norm, MAX pool, PFN dot-sim head), K=5 balanced support graphs/class from the OGB
train split, balanced query cap from the OGB test split, 3 seeds:

```
ogbg-molhiv 5-shot AUC:  seed11=0.6711  seed22=0.5356  seed33=0.5261
  mean = 0.5776 +- 0.0662   (NQ=260/seed)
PAPER reference (molhiv 5-shot): ~0.5817
```

**0.578 vs paper 0.582 — essentially a match.** This validates both the released
checkpoint and our wiring (the loader patches: inject missing `args.degree`, inject
saved head-depth params, then **assert a strict zero-missing/zero-unexpected state
load** — it refuses to score a partially-random model; it did NOT refuse). Further
evidence the checkpoint is genuinely trained, not a placeholder:
`epoch=26`, `best_metrics.gc_test_metric=0.636`, transformer weights with
non-trivial trained statistics (e.g. layernorms at ~0.992, not exactly 1.0).
Smoke log: `/home/scratch/bench_out/gilt_smoke_molhiv.log`.

**Proceeded to the full matched sweep.**

## Environment recipe (clpc35, GPU box)

- GPU box: **clpc35.cs.ox.ac.uk**, RTX 5000 Ada 32GB (sm_89). `/home/scratch` is
  NOT shared with clpc95 — only NFS home `/users/setman` is. All GILT artifacts are
  clpc35-local; results rsynced back to clpc95 for figures.
- venv: reused **`/home/scratch/gpfn_venv`** (torch 2.4.0+cu118). All GILT deps
  already present — torch_geometric 2.8.0, torch_sparse 0.6.18+pt24cu118,
  torch_scatter 2.1.2+pt24cu118, ogb 1.3.6, sklearn 1.9.0. No new installs needed.
- repo: `github.com/yiming421/inductnode` @ `ba46cf4` cloned to
  **`/home/scratch/graphlex_icl/inductnode`**; checkpoint at
  **`/home/scratch/graphlex_icl/inductnode/checkpoints/gilt_model.pt`** (26 MB,
  sha256 `1caebd22…aa70`).
- data: OGB molhiv auto-downloaded to `/home/scratch/ogb`; TUDataset at
  `/home/scratch/tudata`; sweep manifest/splits at
  `/home/scratch/graphlex_icl/manifest.json`.
- Run recipe (smoke gate then sweep):
  ```bash
  ssh clpc35; source /home/scratch/gpfn_venv/bin/activate
  export GILT_REPO=/home/scratch/graphlex_icl/inductnode
  export GILT_CKPT=$GILT_REPO/checkpoints/gilt_model.pt
  export OGB_ROOT=/home/scratch/ogb TU_ROOT=/home/scratch/tudata
  export SWEEP_MANIFEST=/home/scratch/graphlex_icl/manifest.json
  export PYTHONPATH=/home/scratch/graphlex_icl:/home/scratch/graphlex_icl/eval:$PYTHONPATH
  cd $GILT_REPO
  python /home/scratch/graphlex_icl/eval/gilt_smoke_molhiv.py --shots 5 --queries 500 --seeds 11,22,33
  python /home/scratch/graphlex_icl/eval/gilt_icl.py --seeds 11,22,33 --out /home/scratch/bench_out/gilt --force
  ```
  (note: the user's shell has `noclobber`; use `set +o noclobber` before `>` redirects.)

## Matched-split sweep — 30 datasets, 8 sciences, BALANCED accuracy

`eval/gilt_icl.py` over all 30 sweep datasets at the **IDENTICAL** few-shot
splits/seeds the other arms use (5 shots/class, 40 queries, seeds 11/22/33). The
script asserts reconstructed query truth == sweep-manifest truth per dataset/seed
(SystemExit on mismatch) — **all 30 datasets passed parity → provably
apples-to-apples** with graphlex+LLM. GILT featurization: clean one-hot node
categories (same `node_cats` the LLM/logreg arms use) else degree one-hot →
PCA/zero-pad to 128 → L2norm; MAX pool. 30/30 ran cleanly, 0 failures.

Per-domain mean balanced accuracy (macro per-class recall; majority == chance):

| domain | n | chance | classical (logreg) | **GILT** | GraphPFN-embed | graphlex+Opus |
|---|---|---|---|---|---|---|
| chemistry | 9 | 0.500 | 0.596 | 0.516 | 0.619 | **0.626** |
| biology | 2 | 0.333 | 0.386 | **0.425** | 0.353 | 0.424 |
| neuroscience | 3 | 0.500 | 0.514 | 0.482 | **0.520** | 0.478 |
| social | 6 | 0.444 | **0.544** | 0.541 | 0.502 | 0.542 |
| vision | 4 | 0.077 | 0.525 | 0.349 | 0.130 | **0.529** |
| synthetic | 4 | 0.235 | 0.452 | 0.331 | 0.427 | **0.550** |
| citation | 1 | 0.500 | 0.664 | **0.726** | 0.627 | 0.604 |
| archaeology | 1 | 0.033 | **0.219** | 0.191 | 0.136 | 0.194 |
| **OVERALL** | 30 | 0.371 | 0.525 | 0.461 | 0.461 | **0.543** |

Overall mean **regret** (method − best non-LLM baseline = max(classical, majority)):

| method | mean regret | n |
|---|---|---|
| graphlex+Opus | **+0.016** | 30 |
| classical (logreg) | −0.002 | 30 |
| **GILT** | **−0.066** | 30 |
| GraphPFN-embed (frozen+logreg) | −0.065 | 30 |

GILT is above chance on **24/30** datasets, but beats classical-logreg on only
**10/30** (graphlex+Opus beats logreg on 18/30).

### Reading
- **The checkpoint is real and GILT works** (smoke passes, runs end-to-end on all 30
  at matched splits). This is a genuine, validated comparison #2.
- But as a **downstream cross-domain** few-shot whole-graph classifier at this
  low-label budget, GILT is **modest**: overall 0.461 balanced acc, regret −0.066 —
  statistically on par with the GraphPFN-embed arm (0.461 / −0.065) and **behind
  both classical-logreg (0.525) and graphlex+Opus (0.543, the only arm with positive
  regret)**. GILT shines only on a few sets (DBLP_v1/citation 0.726, PROTEINS 0.634)
  and collapses to ~chance on several chemistry/synthetic sets where logreg/Opus do
  fine (e.g. AIDS 0.544 vs Opus 0.949; MUTAG 0.465 vs Opus 0.767; TRIANGLES 0.396 vs
  Opus 1.000). Likely cause: GILT's PCA→128 featurization throws away the explicit
  node-type/structural signal that graphlex's verbalization and logreg-on-facts keep.
- Net for the paper: comparison #2 strengthens the flexibility headline — a
  dedicated, pretrained native-ICL **graph** FM does **not** beat the simple
  classical baseline cross-domain, while graphlex+LLM does (positive regret), with
  no per-task training for either.

## Files written
- `/home/scratch/bench_out/gilt/<dataset>.json` — 30 per-dataset results, keyed
  `model="gilt"` (mirrors fm_repr/gnn keying: spc, n_classes, n_query, seeds, feat,
  results.gilt.{seed:balacc}, mean.gilt=[mu,sd], config). Also mirrored on clpc95.
- `/home/scratch/bench_out/gilt/_gilt_balacc_by_dataset.json` — flat dataset→mean
  balanced-acc map (convenience for figures).
- `/home/scratch/bench_out/gilt_smoke_molhiv.log` — the smoke gate output.
- `/home/scratch/bench_out/gilt_sweep.log` — full sweep log (30 DONE, 0 failed).

## How it folds into `eval/make_figures.py`
GILT is a new **non-LLM FM** column alongside `graphpfn` in the cross-domain regret
heatmap (`fig_regret_heatmap*.png`). Minimal, mechanical hook (no graphlex-core
change):
1. Add a `gilt_balacc(dataset)` reader paralleling `fm_repr_balacc()` — read
   `/home/scratch/bench_out/gilt/<dataset>.json` → `mean.gilt[0]` (None if missing).
2. In `compute_sweep_table()` add `"gilt": gilt_balacc(ds)` to each row (and bump
   the `_sweep_cache.json` cache, or run with `--refresh`).
3. Add `("gilt", "GILT\n(native graph-ICL)")` to the `METHODS` list — place it
   right after `graphpfn` so the two non-LLM FM arms sit together above the LLMs.
   The regret math (`v - max(classic, majority)`) and missing-cell hatching already
   handle it generically.

## Fallback note (NOT attempted in this run)
Per the experiment plan, the alternate whole-graph native-ICL FM is **GILT + a
TabPFN-molprop arm**. GILT passed its gate, so the fallback is **not** needed for
validity. If a stronger chemistry-specific native-ICL comparison is later wanted,
**TabPFN-molprop** is the recommended fallback — but it was deliberately NOT run
here (out of scope for this validation pass).
