# KG tail-prediction track — lab note

Lab note only (no manuscript prose). Sibling of KG_TRACK_PLAN.md / NODE_TRACK_PLAN.md.

## What this track is

The **entity-ranking** variant of the KG suite: **tail prediction** — given a head `h`
and a relation `r`, rank candidate **tail** entities and put the true `t` first. This is
**ULTRA's NATIVE task** (arXiv 2310.04562 is built and pretrained for tail/entity ranking
given a relation), so it is the **home-turf comparison** for the specialist FM foil.

It completes the KG **2×2**:

| | graphlex + LLM | ULTRA (specialist FM) |
|---|---|---|
| **relation prediction** (`kg_icl.py`) — rank the 46 relations for a fixed `(h,t)` | spans it | **off** its objective |
| **tail prediction** (`tail_pred_icl.py`, here) — rank the 135 tails for a fixed `(h,r)` | spans it | **NATIVE** |

graphlex+LLM spans **both** cells in one results table; the specialist FM is strong only in
its home cell (tail-pred) and weak off it (relation-pred, where in the sibling track ULTRA
landed below the tiny DistMult baseline). That is the granularity-lock the suite is about.

## KG chosen

**UMLS** (135 entities, 46 relations, 6 529 triples; readable entity + relation names).
Same `/home/scratch/kg_data/{UMLS,Nations,Kinship}/{train,valid,test}.txt` as the sibling
track. UMLS wins for the smoke: 135 short readable entity names list cleanly as a candidate
menu, and rank metrics over 135 entities are meaningful (chance MRR ≈ 0.0147, chance Hits@10
≈ 0.074). Scale-up to Nations/Kinship is a drop-in (same loader/protocol).

## Protocol (reuses kg_icl.py — does NOT reinvent)

- **Query** = ordered pair `(h, r)`; **label** = the tail `t`. Queries sampled from the
  **test** split (`make_splits`, same de-dup-by-undirected-`(h,t)` rule as kg_icl.py).
- **Few-shot support** = K labeled `(h, r → t)` examples (K ∈ {1, 3}), each verbalized like
  a query (with its true tail shown as `[answer: E… = name]`).
- **Candidate-entity menu.** All 135 UMLS entities → tokens `E000..E134` (zero-padded to
  vocab width) with readable names, listed up front: `E041 = drug_delivery_device`, …. The
  LLM answers with these tokens.
- **Verbalization (graphlex angle).** For the **head** `h` we verbalize the **untyped
  structural skeleton** of `h`'s 1-hop neighborhood via `graphlex.facts()` /
  `verbalize(focus='structure')` (graphlex renders untyped structure), then **APPEND** a
  readable **typed-triples context line** — the observed triples incident to `h`. The query
  triple (by undirected `(h,t)`) is **never shown**. This is exactly kg_icl.py's append
  pattern (`graphlex core is NOT modified`); reuses `joint_nx` (called with `h==h` to get
  `h`'s k-hop ball). The typed-context line is the load-bearing signal; the structural
  skeleton is the graphlex framing.
- **Output format.** One line per query, `<id> <t1,t2,…,t10>` — the model's **top-10
  ranked** entity tokens, best first, comma-separated. A **new ranked-output parser**
  (`score_tail_pred.parse_ranked`, extending the `_common.ANS_LINE` tolerance: handles
  `Query N`, punctuation, comma/space separators, surrounding chatter) extracts the
  per-query ranking. A truth absent from the top-10 → rank ∞ (MRR 0, no hit) — the honest
  treatment of a top-K answer. Driver-compatible `seed*_k*.txt` filenames + manifest.
- **Leakage control.** Per seed, the query + support `(h,t)` pairs (undirected) are removed
  from the observed graph used for (a) verbalization, (b) the typed-context line, and (c)
  training the DistMult baseline — same `observed_triples` rule as kg_icl.py.
- **Filtered ranking** (standard KG link-prediction eval). For a query `(h, r)`, *other*
  true tails of `(h, r)` (across the full graph) are removed from the ranking before
  locating the held-out truth, so a model is not penalised for ranking another correct tail
  above the held-out one. Stored per query in the manifest as `filter_tails`; applied
  identically to DistMult, the LLM arms, and ULTRA.

## Metrics

Rank-based, standard for KG link prediction: **Hits@1, Hits@10, MRR** of the true tail
(filtered). **Primary = MRR + Hits@1**; Hits@1 is the accuracy-like number (balanced
accuracy is meaningless over 135 entities). Mean ± std over seeds.

## Baseline matrix

| arm | what | granularity | status |
|-----|------|-------------|--------|
| **freq-prior** | rank tails by how often they are the tail of relation `r` in the observed graph (ties / unseen → global tail freq → id). A strong, standard KG baseline | KG | RUN |
| **DistMult (KG-emb)** | numpy DistMult `score(h,r,t)=Σ e_h·w_r·e_t`, tiny epoch budget (dim 64, 300 ep, lr 0.05); tail-pred = rank ALL 135 tails by `score(h,r,e)`; trained on observed (leakage-stripped) triples only. **Reuses `kg_icl.train_distmult` verbatim** | KG | RUN |
| **graphlex + LLM** | head-neighborhood verbalization → Qwen-2.5-14B / Opus, K-shot ICL, ranked top-10 | **none** (spans all granularities) | RUN (smoke) |
| **ULTRA (zero-shot)** | the FM foil, **NATIVE** task | KG entity ranking | **ENV-PENDING** (matched-run spec below; tail-mode hook implemented, not run) |

### ULTRA matched-run spec (ENV-PENDING — ready to run on clpc35)

ULTRA is already set up on clpc35 (`/home/scratch/gpfn_venv`, `ultra_4g.pth`, repo at
`/home/scratch/ultra_work/ULTRA`). Its UMLS **tail-ranking smoke already passed**
(MRR 0.275 / Hits@1 0.160 / Hits@10 0.550 over all 135 entities — ~19× chance MRR), so the
checkpoint does sane zero-shot inductive tail ranking on UMLS. Tail prediction is exactly
what `ultra_kg.smoke()` already does; the matched run reuses that machinery over **our**
queries.

**Tail-mode hook (implemented, NOT run).** `eval/ultra_kg.py` (clpc35-local copy at
`/home/scratch/ultra_work/ultra_kg.py`) now has a **`tail` mode** (`run_tail()`), spliced in
parallel to `run()` and syntax-checked on clpc35. It:
1. reads `/home/scratch/bench_out/tail_pred_icl/UMLS/manifest.json` (task `tail_prediction`);
2. per seed reads that seed's `query_triples` + `support_triples`, strips those `(h,t)`
   pairs (undirected) from the observed graph — **identical leakage rule** to DistMult / the
   LLM arms — and builds the ULTRA inference `Data` (`make_data`, doubled inverse relations,
   relation graph), **the same seeds {11,22} / NQ=12 query pairs**;
3. for each query `(h, r)`: ranks the true tail among **all 135 entities** by ULTRA's
   `(h, t, r)` triple score (one batch over all candidate tails — the same call as the
   smoke), applies the **filtered** setting (`filter_tails` → other true tails set to −∞),
   `rank` → Hits@1 / Hits@10 / MRR;
4. writes `tail_result.json` + splices an `"ultra"` block into the tail-pred manifest
   (parallel to `"distmult"`); `score_tail_pred.py` already prints an `ULTRA (native)
   ENV-PENDING` row that drops in from that block.

**To run (next step; GPU, ~free of LLM tokens):**
```
# /home/scratch is clpc35-LOCAL (not shared). rsync the manifest over first:
rsync -a /home/scratch/bench_out/tail_pred_icl/ clpc35:/home/scratch/bench_out/tail_pred_icl/
ssh clpc35
PATH=/home/scratch/gpfn_venv/bin:/usr/local/cuda-13.2/bin:$PATH \
  CUDA_HOME=/usr/local/cuda-13.2 CUDA_VISIBLE_DEVICES=0 \
  /home/scratch/gpfn_venv/bin/python /home/scratch/ultra_work/ultra_kg.py tail UMLS
# then rsync the manifest/tail_result.json back and re-score:
rsync -a clpc35:/home/scratch/bench_out/tail_pred_icl/UMLS/ /home/scratch/bench_out/tail_pred_icl/UMLS/
python eval/score_tail_pred.py UMLS
```
**Matched-comparison guarantee:** ULTRA and the LLM score the **same** query `(h,r)` pairs,
same seeds, same leakage-stripped observed graph, same filtered setting — because both read
the **one** manifest written by `tail_pred_icl.py`. Do **not** run on clpc35 in this smoke
(leave the GPU run as the next step). On its native task ULTRA is **expected to be strong**
here (the on-message half of the 2×2).

## Token budget (this smoke — STAYED IN BUDGET)

SMOKE only. UMLS, 2 seeds {11,22}, K ∈ {1,3}, NQ=12, 1 readable rep → **4 prompt files**.
Each prompt ≈ 36.7 K chars ≈ **9.2 K tokens** (dominated by the 135-entity menu + 12
verbalized queries). Non-LLM baselines (DistMult + freq-prior) are free.
- **Qwen-2.5-14B**: 4 calls (free, local on clpc35 via `run_qwen.py`, `NUM_CTX=16384`).
- **Opus**: **exactly 4 calls** (one per prompt file), ≈ **9.2 K input tokens each ≈ 37 K
  input total**, + ~160 output tokens each (the 4 × 12 ranked lines). Total Opus spend
  ≈ **37–38 K tokens** — well under the 100 K cap. (Run via a tiny inline driver identical
  to `run_opus_cli.py` except the trailing format instruction asks for the ranked
  `<id> <t1,…,t10>` form instead of `<id> <CLASS>`.)

## Smoke results (UMLS, 2 seeds, FILTERED ranking; chance MRR 0.0147 / Hits@10 0.074)

| arm | Hits@1 | Hits@10 | MRR |
|-----|-------:|--------:|----:|
| freq-prior | 0.708 ± 0.125 | 0.917 ± 0.083 | 0.792 ± 0.104 |
| DistMult (held-out, leakage-stripped) | 0.708 ± 0.042 | 0.958 ± 0.042 | 0.800 ± 0.008 |
| **graphlex + Opus, K=1** | **0.833 ± 0.083** | 0.875 ± 0.042 | **0.854 ± 0.062** |
| graphlex + Opus, K=3 | 0.792 ± 0.042 | 0.833 ± 0.083 | 0.812 ± 0.062 |
| graphlex + Qwen-14B, K=1 | 0.042 ± 0.042 | 0.250 ± 0.167 | 0.115 ± 0.031 |
| graphlex + Qwen-14B, K=3 | 0.083 ± 0.083 | 0.208 ± 0.208 | 0.123 ± 0.123 |
| ULTRA `ultra_4g` (zero-shot, NATIVE) | ENV-PENDING | ENV-PENDING | ENV-PENDING |

Reading (honest, smoke-level):
- **Opus** uses the appended typed-context line + neighborhood framing effectively: best
  **Hits@1 (0.833)** and **MRR (0.854)** of any arm, edging the free baselines. (Its Hits@10
  is bounded by the **top-10** answer length — it returns exactly 10 candidates — so Hits@10
  ≤ Hits@10 of a full-ranking baseline by construction; Hits@1/MRR are the fair head-to-head.)
- **DistMult ≈ freq-prior** are both strong on UMLS because UMLS relations are heavily
  **type-constrained** — a handful of tails dominate each relation, so a frequency prior is
  hard to beat. This is the honest, strong KG bar (and explains their high Hits@10).
- **Qwen-14B collapses** (Hits@1 ≈ 0.04, repeats a generic candidate list across queries) —
  same capability gap as the relation-pred sibling, not a plumbing bug (answers parse cleanly
  as ranked lists; the parser is exercised and correct).
- This is a **smoke** (2 seeds, 12 queries, top-10 cap). Scale-up before any claim: 3 seeds,
  NQ ≥ 30, ask top-20 (to fairly measure Hits@10 vs full-ranking baselines), add
  Nations/Kinship, and **run the ULTRA row** (its native task — expected strong).

## How to run / scale up

```
# generator + baselines (no LLM):
SMOKE=1 python eval/tail_pred_icl.py UMLS    # 2 seeds, K{1,3}, NQ=12 (the smoke)
python eval/tail_pred_icl.py UMLS            # full: 3 seeds, K{1,3}, NQ=20
# LLM arms (drivers unchanged; Opus driver's trailing instruction adjusted to the ranked form):
NUM_CTX=16384 STRICT=1 python eval/run_qwen.py /home/scratch/bench_out/tail_pred_icl/UMLS
python eval/run_opus_cli.py /home/scratch/bench_out/tail_pred_icl/UMLS   # (top-10 ranked output)
# score:
python eval/score_tail_pred.py UMLS
# ULTRA (clpc35, next step): see matched-run spec above.
```

Files: `eval/tail_pred_icl.py` (generator + DistMult/freq baselines, no LLM),
`eval/score_tail_pred.py` (ranked-output parser + Hits@1/Hits@10/MRR scorer, reuses
`_common` style), prompts/manifest under `/home/scratch/bench_out/tail_pred_icl/UMLS/`.
ULTRA tail-mode hook in `eval/ultra_kg.py` (clpc35-local) — implemented, syntax-checked,
**not run** (GPU run = the next step). Nothing committed.
```
