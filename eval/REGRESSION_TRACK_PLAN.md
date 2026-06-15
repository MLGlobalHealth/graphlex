# REGRESSION_TRACK_PLAN — graph-level CONTINUOUS-target ICL for graphlex+LLM

Lab note. Scoping + design + smoke results for a **graph-level REGRESSION** in-context
-learning (ICL) track: predict a **continuous** number per graph, the cell of the
taxonomy the other tracks don't cover (they are all *categorical*).

Harness is implemented and smoke-tested: `eval/regression_icl.py` (generator +
free baselines) + `eval/score_regression_icl.py` (numeric parser + MAE/RMSE/R²
scorer). This doc records the design + the smoke numbers; the code is the deliverable.

## 1. Motivation — completing the granularity × output-type grid

The strategic point of the suite so far is **task-granularity-agnosticism**: the same
`facts()` / `verbalize()` machinery does whole-graph classification (`sweep.py`,
`label_curve.py`), node classification (`node_icl.py`), and edge/link prediction
(`edge_icl.py`). But all three predict a **class token**. This track adds the missing
*output type* — a **continuous** target — so graphlex+LLM spans the full grid:

|                         | classification                   | **regression**          |
|-------------------------|----------------------------------|-------------------------|
| **whole-graph**         | sweep.py / label_curve.py        | **regression_icl.py** ← |
| **node**                | node_icl.py                      | (future: node regress)  |
| **edge**                | edge_icl.py                      | (future: edge weight)   |

Nothing about the method changes — only *what the prompt asks* (emit a real number
instead of a class) and *how it is scored* (MAE/RMSE/R² instead of balanced accuracy).
One prompt change, one scorer change. Graph foundation models are mostly locked to
classification heads; native few-shot graph **regression** ICL is rare (GILT/TabPFN-
molprop is the one whole-graph-regression FM candidate — see the graphfm-icl-landscape
memo). That is the table this track sets up.

## 2. Dataset — MoleculeNet FreeSolv (smallest sensible continuous target)

`torch_geometric.datasets.MoleculeNet`. Picked the **smallest** so the smoke is cheap
and the full table is fast. FreeSolv chosen; ESOL / Lipophilicity wired as one CLI arg.

| dataset (MolNet) | graphs | target                              | mean ± std (raw)  |
|------------------|:------:|-------------------------------------|-------------------|
| **FreeSolv**     | 642    | hydration free energy (kcal/mol)    | −3.80 ± 3.84      |
| ESOL             | 1128   | log-solubility (log mol/L)          | (one CLI arg)     |
| Lipophilicity    | 4200   | octanol/water logD                  | (one CLI arg)     |

`DATASETS` dict in `regression_icl.py` maps each to its MolNet name + human target.
Molecules are tiny (FreeSolv mean 8.7 atoms, max 24) → whole molecule fits in context;
no ego-graph trick needed (unlike the node/edge tracks). **rdkit** is required by the
PyG MoleculeNet SMILES→graph processor and was `pip install`ed into the venv
(`rdkit-2026.3.3`); raw CSV cached at `/home/scratch/molnet/freesolv/raw/SAMPL.csv`,
processed dataset at `/home/scratch/molnet/freesolv/`.

## 3. Verbalization — molecule → graphlex facts (core untouched)

Each molecule → undirected `networkx` graph with a `'type'` node attribute = **element
symbol** (from MoleculeNet atom-feature column 0 = atomic number; `Z2SYM` maps
6→C, 7→N, 8→O, 9→F, 15→P, 16→S, 17→Cl, 35→Br, 53→I — the 9 elements in FreeSolv) →
`verbalize(facts(G, node_attrs='type'), focus='structure')`. A readable **element-
composition line** (`Atoms (13 total): Cx10, Ox3, ...`) is appended after the structure
block — exactly parallel to how `node_icl.py` appends a per-node readable line.
**graphlex core is NOT modified.** This is the SAME `facts()`/`verbalize()` the LLM
sees and that feeds the Ridge baseline (`_common.fvec`), so the classical bar uses the
identical features.

## 4. Few-shot regression protocol + the standardization question

- **K-shot support:** `K_SHOTS = [1, 3, 5, 10]` example molecules (graph → value),
  nested in K (K=1 ⊂ K=3 ⊂ …). `NQ = 12` query molecules, disjoint from shots.
  `make_splits` draws them from a per-seed permutation, plus a `TRAIN_FULL = 400`
  pool (disjoint from queries) for the full-supervision GNN upper bar.
- **Seeds:** `SEEDS = [11, 22, 33]` (≥3, load-bearing).
- **TARGET STANDARDIZATION (load-bearing design choice, documented):** the prompt
  shows targets as **z-scores** `z = (raw − zmean) / zstd`, so the numbers the LLM
  emits are O(1) and unit-free (typically [−3, 3]) — far easier for an LLM than raw
  kcal/mol. `zmean`/`zstd` are the **K-shot** mean/std and are **stated in the prompt
  header** so they can be inverted at scoring time. **K=1 fix:** a single shot has
  std≈0, which makes the z-score divisor meaningless — so when the shot std ≤ 1e-3 we
  fall back to the **global target std** as the scale (recorded in the manifest per
  file as `zmean`/`zstd`; the scorer uses the *same* values to de-standardize). The
  scorer multiplies the LLM's z-prediction by `zstd` and adds `zmean` to recover raw
  kcal/mol **before** computing MAE/RMSE/R², so the LLM and the baselines are all
  reported in the **same raw units**. Baselines are computed on the **raw** target
  directly (no standardization needed for Ridge/GNN/mean).
- **Output format:** `'<id> <number>'` (a real z-score, e.g. `0 -0.73`). The existing
  drivers (`run_qwen.py`, `run_opus_cli.py`) are used **unchanged** — they just pipe
  the prompt and capture stdout; the format difference (number vs class token) is
  entirely handled in the scorer's numeric parser.

## 5. Baseline matrix (all FREE — no LLM — RUN end-to-end)

| baseline                         | status              | how                                                                 |
|----------------------------------|---------------------|---------------------------------------------------------------------|
| **predict-the-mean** (floor)     | **run**             | `mean_at` — predict the K-shot mean for every query; trivial floor   |
| **Ridge on graphlex fact-vector**| **run**             | `ridge_at` — `_common.fvec(facts(G))`, StandardScaler+Ridge(α=1), SAME features the LLM sees, SAME K shots (K=1 degenerates → predicts the lone shot) |
| **GNN regressor, few-shot**      | **run**             | `gnn_at` — GIN(3 layers)+global-mean-pool+linear head, MSE, trained on the K shots only; near-useless at low K (reported honestly) |
| **GNN regressor, full-supervision** | **run (upper bar)** | `gnn_at` on the `TRAIN_FULL=400` pool, K-independent — the specialist ceiling, parallel to the node-track transductive GCN |
| graphlex+LLM (Opus / Qwen)       | prompts ready / smoke run | `<DS>/facts/seed*_k*.txt` → driven by `run_qwen.py`/`run_opus_cli.py` → `ans/<model>/` |

All baselines written into `manifest.json` as `{metric: [per-seed vals]}` blocks
(`ridge`, `mean_baseline`, `gnn_fewshot`, `gnn_full`). Ridge + mean are
representation-independent; the GNN trains on the raw atom-feature matrix.

### 5b. GNN regressor config (`regression_icl.py::GINReg` / `gnn_at`)

GIN encoder (3× `GINConv`, hidden 64) + `global_mean_pool` + a linear head → scalar,
MSE loss, target standardized per train pool (de-standardized at eval). Adam lr 1e-2,
wd 5e-4, ≤200 epochs, early-stop patience 30 on train-loss plateau. CPU, seconds per
fit. The few-shot variant (train on K) is expected weak at K=1..few; the full-
supervision variant (train on 400) is the honest specialist upper bar.

## 6. Metrics — regression, NOT balanced accuracy

`reg_metrics` (same fn in generator + scorer): **MAE**, **RMSE**, **R²** (1 − SSE/SST),
mean over seeds, all in **raw target units (kcal/mol)**. R² < 0 means "worse than
predicting the query-set mean" — common and honest at tiny K. The **numeric parser**
(`score_regression_icl.py::parse_num`) is tolerant (mirrors `_common.parse_ans` for
class tokens, but for numbers): pulls the id, then takes the **last signed float** on
the line, surviving `"0 -0.73"`, `"Query 1: 1.2"`, `"2) 0.5"`, `"3 - -0.45"`,
`"4: z=-1.4e0"`, `"6 0.33 kcal/mol"`, unicode-minus, tabs. Verified against a fixture.

## 7. SMOKE — what was actually run, and the token budget

**Free baselines: run FULLY** — FreeSolv, 3 seeds, K∈{1,3,5,10}, NQ=12. Cost: CPU,
~minutes. **LLM smoke: deliberately tiny** — ONE dataset (FreeSolv), ONE seed (seed11),
k∈{1,3}, NQ=12 → **2 prompts per model**. Prompts are ~1.5K (k=1) to ~2.4K (k=10)
tokens — well under the 10K cap. The other 10 prompt files were moved aside during the
smoke so the (unchanged) drivers globbed only the 2 smoke prompts, then restored.

- **Qwen (free, local, primary):** `qwen2.5:14b-instruct` on clpc35 via `run_qwen.py`
  — 2 calls. Emitted clean `<id> <number>` z-scores, no format coaxing.
- **Opus (paid, confirmation only):** `run_opus_cli.py` via the local `claude -p` CLI —
  **EXACTLY 2 Opus calls** (k=1, k=3; seed11). Each prompt ≈ 1.5–1.7K input tokens +
  ~12 output lines ≈ a few hundred output tokens → **well under ~5K tokens total Opus
  spend** for the whole task (budget was «100K). Both parsed cleanly and gave sane,
  varied numbers (more spread than Qwen).

## 8. Smoke results (FreeSolv, raw units kcal/mol)

**Free baselines (3 seeds, mean):**

| K   | predict-mean (floor) | Ridge (facts)        | GNN few-shot          |
|:---:|----------------------|----------------------|-----------------------|
| 1   | MAE 9.27 R² −6.36    | MAE 9.27 R² −6.36*   | —                     |
| 3   | MAE 3.82 R² −1.14    | MAE 8.04 R² −7.34    | MAE 6.77 R² −8.07     |
| 5   | MAE 3.98 R² −0.94    | MAE 7.89 R² −6.57    | MAE 4.62 R² −1.89     |
| 10  | MAE 2.72 R² −0.16    | MAE 5.62 R² −3.63    | MAE 2.89 R² −0.21     |
| **GNN full-supervision (train 400, K-independent UPPER BAR)** | | **MAE 1.74 RMSE 2.24 R² 0.595** |

\*K=1 Ridge degenerates to predicting the single shot value (no fit possible) → equals
the K=1 mean baseline by construction.

**graphlex+LLM smoke (seed11 only, de-standardized to raw units, cov = #queries parsed):**

| K   | Opus                          | Qwen                              |
|:---:|-------------------------------|-----------------------------------|
| 1   | MAE 5.66 R² −4.16 (cov 12)    | MAE 11.61 R² −17.35 (cov 12)      |
| 3   | **MAE 2.84 R² −0.70 (cov 12)**| MAE 3.44 R² −0.81 (cov 12)        |

**Reading (smoke, single seed — directional only):** at K=3, **Opus (MAE 2.84) already
beats predict-the-mean (3.82), crushes few-shot GNN (6.77) and Ridge-on-facts (8.04),
and approaches the full-supervision GNN ceiling (1.74)** — purely from verbalized
structure + element composition, zero molecular features. That is exactly the
granularity/output-type-flexibility story this track exists to tell. K=1 is noisy for
both LLMs (a single shot anchors the z-scale poorly). Qwen tracks the same shape but
weaker. **Caveat:** one seed, 12 queries — a smoke, not a verdict. The full pass
(3 seeds × K∈{1,3,5,10}, both models) is one driver run away; the baselines for it are
already in the manifest.

## 9. What is ready for the full LLM pass

- **12 prompt files** written: `bench_out/regression_icl/FreeSolv/facts/seed*_k*.txt`
  (3 seeds × 4 K). Same skeleton as every track (TASK → `=== LABELED EXAMPLES ===`
  with `[value z=…]` blocks → `=== QUERIES ===` → `OUTPUT FORMAT: '<id> <number>'`).
- Run `run_qwen.py` / `run_opus_cli.py` over `bench_out/regression_icl/FreeSolv`
  (they glob `*/seed*_k*.txt`) → answers to `facts/ans/<model>/seed*_k*.ans` → score
  with `score_regression_icl.py FreeSolv`. ESOL / Lipophilicity are one CLI arg away.
- Manifest carries per-file `support_ids`, `query_ids`, `zmean`, `zstd`, `truth`
  (raw values) so a native-regression FM (e.g. GILT/TabPFN-molprop, the whole-graph-
  regression FM candidate) could consume the identical splits and write
  `ans/<fm>/…` for the same scorer.

## 10. Decisions — RESOLVED / OPEN

1. **Dataset.** RESOLVED: **FreeSolv** (smallest, 642 graphs). ESOL/Lipo = one arg.
2. **Verbalization features.** RESOLVED: structural `facts()` + element-composition
   line, `node_attrs='type'` = element symbol. (Could add bond-type / degree-sequence
   verbalization later; molecular descriptors would be the "readable rep" analog.)
3. **Standardization.** RESOLVED: per-(seed,K) z-score, global-std fallback at K=1,
   inverted in the scorer (§4). Documented in the manifest.
4. **Trivial floor + specialist ceiling.** RESOLVED: predict-the-mean floor + full-
   supervision GNN ceiling both reported (the few-shot GNN alone is near-useless at low
   K, as expected — so the upper bar is essential to bracket the LLM honestly).
5. **OPEN — full pass / more datasets / more seeds.** Smoke is single-seed; run all
   3 seeds × 4 K on FreeSolv (+ ESOL) for the table. Cost is flat (tiny molecules).
6. **OPEN — node/edge regression.** This track is whole-graph regression; node-level
   (e.g. atom property) and edge-level (bond/weight) regression would fill the rest of
   the regression column, reusing this scorer + standardization machinery verbatim.
