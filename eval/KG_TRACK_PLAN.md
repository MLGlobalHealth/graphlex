# KG relation-prediction track — lab note

Lab note only (no manuscript prose). Sibling of NODE_TRACK_PLAN.md / GNN_BASELINE_PLAN.md.

## What this track is

The **multi-relational / typed-edge** variant of the edge track. The whole-graph
(`sweep.py`), node (`node_icl.py`), and edge/link (`edge_icl.py`) tracks all ask
yes/no-or-class questions about **untyped** structure. Here the edges carry **types**:
the graph is a set of `(head, relation, tail)` triples over a relation vocabulary `R`.

**Task = RELATION PREDICTION** (chosen over triple classification — cleaner multi-class,
single balanced-accuracy number). Given an ordered pair `(h, t)` that is **known to be
linked**, predict which relation `r` connects `HEAD -[r]-> TAIL`, as a classification
over `R`. This is **ULTRA's home turf** (arXiv 2310.04562) and extends the
granularity-flexibility story to typed edges: graphlex+LLM still spans the task in one
results table, while each specialist graph FM is locked to its granularity (ULTRA = KG
link/relation; GraphPFN = node; PRODIGY/OFA = node).

This is the FOURTH granularity in the suite (whole-graph → node → edge/link →
typed-edge/relation).

## KG chosen

**UMLS** (the classic tiny biomedical KG). Downloaded as text triples to
`/home/scratch/kg_data/{UMLS,Nations,Kinship}/{train,valid,test}.txt`
(source: github.com/ZhenfengLei/KGDatasets, tab-separated `h\tr\tt`).

| KG       | entities | relations | triples | notes |
|----------|---------:|----------:|--------:|-------|
| **UMLS** |      135 |        46 |   6 529 | **chosen** — readable entity+relation names, rich 46-way relation vocab, tiny |
| Nations  |       14 |        55 |   1 992 | scale-down option (very few entities) |
| Kinship  |      104 |        25 |  10 686 | scale-up option (more triples) |

UMLS wins for the smoke: entities (`acquired_abnormality`, `eicosanoid`, …) and
relations (`location_of`, `isa`, `process_of`, …) are **human-readable**, so the typed
neighborhood verbalizes into natural language — ideal for in-context relation prediction
— and 46 relation classes make balanced accuracy meaningful (chance floor 1/46 ≈ 0.022).

**pykeen is NOT installed** in `/home/scratch/fmsn-dev/.venv`; deliberately not installed
(token/time budget). The text-triple files + numpy DistMult are self-contained.
Scale-up to **FB15k-237 / WN18RR** is a drop-in (same loader, same protocol) but too big
for the in-context LLM arm — flagged as a scale-up, not run.

## Protocol

- A **query** = an ordered pair `(h, t)` known to be linked; label = relation `r`
  (multi-class over `R`). Queries sampled from the **test** split.
- Relations → CLASS tokens `R00..R45`, so `_common.parse_ans` + `_common.bal_acc` + the
  existing drivers (`run_qwen.py`, `run_opus_cli.py`) work **unchanged**. The prompt lists
  the candidate-relation menu (`R03 = assesses_effect_of`, …) up front.
- **Few-shot support** = K labeled `(h, t → r)` example pairs (K ∈ {1, 3}), each
  verbalized like a query.
- **Verbalization (graphlex angle):** for a pair we verbalize the **untyped structural
  skeleton** of the joint 1-hop neighborhood via `graphlex.facts()` /
  `verbalize(focus='structure')` (graphlex renders untyped structure), then **APPEND** a
  readable **typed-triples context line** — the observed `(h, rel, x)` / `(x, rel, t)`
  triples around the pair. This is exactly how `edge_icl.py` appends its computed
  link-feature line. **graphlex core is NOT modified.** The typed-context line is the
  load-bearing signal; the structural skeleton is the graphlex contribution that frames it.
- **Leakage control:** the query triple and the support query triples are removed from the
  observed graph used for (a) verbalization, (b) the typed-context line, and (c) training
  the DistMult baseline. Removal is by **undirected `(h,t)` pair**, so neither the query
  relation nor a parallel relation on the same pair can be read off.

## Metric

**Balanced accuracy** over the relation classes (primary, `_common.bal_acc`, consistent
with the whole suite) + **Hits@1 / MRR** for the ranking baseline (DistMult). Mean ± std
over seeds.

## Baseline matrix

| arm | what | granularity-locked? | status |
|-----|------|--------------------|--------|
| **freq-prior** | predict the globally most-common relation | n/a | RUN (floor) |
| **DistMult (KG-emb)** | numpy DistMult `score(h,r,t)=Σ e_h·w_r·e_t`, tiny epoch budget; relation prediction = `argmax_r` / rank over `R`; trained on observed triples only | KG (typed edge) | RUN |
| **graphlex + LLM** | typed-neighborhood verbalization → Qwen-2.5-14B / Opus, K-shot ICL | **none** (spans all 4 granularities) | RUN (smoke) |
| **ULTRA (zero-shot)** | the FM foil | KG link/relation | **ENV-PENDING** (see below) |

DistMult hyperparameters (`dim=64, epochs=300, lr=0.05`) picked by a quick sweep on UMLS
train-triple relation recovery (knee: bal-acc ~0.5, Hits@1 ~0.48, MRR ~0.65). On the
leakage-correct **held-out query pairs** (query pairs removed from training) it scores
lower, as expected — that is the honest bar.

### ULTRA slot-in (ENV-PENDING)

ULTRA (github.com/DeepGraphLearning/ULTRA, public checkpoints `ultra_3g` / `ultra_4g` /
`ultra_50g`) is the foundation-model foil: a **single pretrained** model that does
**zero-shot inductive** KG reasoning on *unseen* KGs by conditioning on the relation graph
(relation-of-relations), so it transfers to UMLS with no UMLS training.

Slot-in (same query pairs, same balanced-accuracy + Hits@1/MRR metric, identical leakage
split):
1. CUDA env on clpc35 (ULTRA needs `torch` + `torch_geometric` + a GPU). Clone ULTRA,
   `pip install -r requirements.txt`, download a public checkpoint.
2. Wrap UMLS as an ULTRA `Dataset` (entity/relation dicts already built by `load_kg`;
   feed ULTRA the **observed** triples as the inference graph — the same leakage-stripped
   graph the other arms see).
3. For each query pair `(h, t)`: ULTRA natively ranks **tails given `(h, r)`**; for
   **relation** prediction, score each candidate `r` by ULTRA's `(h, r, t)` triple score
   and take `argmax_r` / the rank of the true `r`. Report balanced-acc + Hits@1 + MRR
   over the identical query set.
4. Write into `manifest.json` under an `"ultra"` block parallel to `"distmult"`;
   `score_kg_icl.py` already prints an `ULTRA (zero-shot) ENV-PENDING` row to drop it in.

Do **not** install ULTRA now (token/CUDA budget) — this is the spec.

## Token budget (this smoke)

SMOKE only. UMLS, 2 seeds {11,22}, K ∈ {1,3}, NQ=12 query pairs, 1 readable rep → 4 prompt
files. Each prompt ≈ 30 K chars ≈ 7.7 K tokens (dominated by the 46-relation menu + 12
verbalized queries). Non-LLM baselines (DistMult + freq-prior) are free.
- **Qwen-2.5-14B**: 4 calls (free, local on clpc35 via `run_qwen.py`, `NUM_CTX=16384`).
- **Opus**: **exactly 4 calls** (`run_opus_cli.py`), ≈ **31 K input tokens + ~80 output
  tokens total** — well under the 100 K cap.

## Smoke results (UMLS, 2 seeds, balanced accuracy; chance 0.022)

| arm | bal-acc | Hits@1 | MRR |
|-----|--------:|-------:|----:|
| freq-prior | 0.111 ± 0.000 | — | — |
| DistMult (held-out query pairs) | 0.417 ± 0.083 | 0.417 ± 0.083 | 0.516 ± 0.058 |
| graphlex+Qwen-14B, K=1 | 0.139 ± 0.083 | — | — |
| graphlex+Qwen-14B, K=3 | 0.222 ± 0.000 | — | — |
| graphlex+Opus, K=1 | **0.722 ± 0.111** | — | — |
| graphlex+Opus, K=3 | 0.611 ± 0.000 | — | — |

Reading: **Opus** uses the appended typed-context line effectively and clears DistMult by
a wide margin on relation prediction; the **14B Qwen** collapses onto a few generic
relations (`causes`, `part_of`) in this 46-way task — a real capability gap, not a
plumbing bug (its answers parse cleanly). This is a smoke (2 seeds, 12 queries); scale-up
= 3 seeds, NQ≥30, add Nations/Kinship, and the ULTRA row, before any claim.

## How to run / scale up

```
# generator + baselines (no LLM):
SMOKE=1 python eval/kg_icl.py UMLS          # 2 seeds, K{1,3}, NQ=12 (the smoke)
python eval/kg_icl.py UMLS                  # full: 3 seeds, K{1,3}, NQ=20
# LLM arms (drivers unchanged):
NUM_CTX=16384 STRICT=1 python eval/run_qwen.py /home/scratch/bench_out/kg_icl/UMLS
python eval/run_opus_cli.py /home/scratch/bench_out/kg_icl/UMLS
# score:
python eval/score_kg_icl.py UMLS
```

Files: `eval/kg_icl.py` (generator + DistMult/freq baselines, no LLM),
`eval/score_kg_icl.py` (scorer, reuses `_common`), prompts/manifest under
`/home/scratch/bench_out/kg_icl/UMLS/`.
