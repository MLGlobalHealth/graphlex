# FUSION_SMOKE — killer-app lab note (Hetionet Compound-treats-Disease)

Lab note only. NO manuscript prose. Design + a token-frugal signal-detection smoke.

## Hypothesis (the "could-move-the-paper" claim)
In a real biomedical KG, entity NAMES carry huge prior knowledge an LLM already has
(drugs like Doxorubicin, diseases like "breast cancer"). A structure-only GNN throws
that identity away; a knowledge-only LLM ignores topology. **If graphlex+LLM (structure
+ names) beats BOTH structure-alone AND knowledge-alone, that is a capability classical
methods categorically lack.** This smoke only asks: *is that gap there at all?*

## Dataset
**Hetionet v1.0 Compound-treats-Disease (CtD)** — `kg_data/hetionet/CtD.tsv`.
- 755 edges, 387 compounds (DrugBank names: Doxorubicin, Aspirin, Prednisone, ...),
  77 diseases (Disease-Ontology names: "breast cancer", "hypertension", ...).
- Small, real, fully human/LLM-readable (the knowledge arm is only meaningful with
  recognizable names — coded-ID KGs like ogbl-biokg are deliberately avoided).
- Bipartite (compounds vs diseases); single relation "treats". Built from the public
  het.io node/edge TSVs (`hetionet-v1.0-nodes.tsv` + `-edges.sif.gz`, metaedge `CtD`).

## Task
Binary link prediction: "does (compound, TREATS, disease) exist?" `CLASS1`=TREATS,
`CLASS0`=NO-TREAT. Balanced held-out positives + sampled true NON-edges (both endpoints
degree>0 so neighborhoods are non-empty). Few-shot K class-balanced labeled examples.
Metric: **balanced accuracy** (`_common.bal_acc`, primary) + AUC for the ranking
baseline. Chance = 0.500.

## The three arms (prompt CONDITIONS over the SAME query edges; identical truth)
1. **knowledge-only** — entity NAMES, NO structure/neighborhood. "Is <drug> a treatment
   for <disease>?" + K name-only labeled examples. Pure parametric recall. ALSO the
   **memorization gauge**: if near-ceiling alone, the task is memorized → temporal
   holdout needed.
2. **anon-structure** — graphlex verbalizes the query edge's joint 1-hop neighborhood
   (reusing the `edge_icl.py`/`kg_icl.py` joint-neighborhood machinery: `joint_nx` →
   `facts()` → `verbalize(focus='structure')` + an appended typed-context edge list),
   but ALL entities are ANONYMIZED (A00=head, A01=tail, A02.. neighbors). Topology
   without identity. ALSO the anonymization control. (Anonymization is total: even
   context-only neighbors beyond the capped neighborhood get fresh A-tokens, verified by
   grepping the anon prompts for `Compound::`/`Disease::` → none.)
3. **fusion** — graphlex neighborhood + REAL entity names. Both signals.

Plus a **non-LLM structural reference**: classical link-prediction heuristics
(common-neighbors / Jaccard / Adamic-Adar / resource-allocation / preferential-
attachment / shortest-path) computed on the leakage-stripped bipartite graph, fed to
logreg on the K support pairs. This is the structure-only classical bar.

## Leakage control
Positive QUERY edges + positive SUPPORT edges are REMOVED (undirected pair set, same
discipline as `edge_icl.py`/`kg_icl.py`) from the observed graph used to (a) build the
verbalized neighborhoods and (b) train/feature the non-LLM baseline. The h–t edge is
also stripped from the joint neighborhood, so the query link is never shown
structurally. No method sees the edge it must predict.

### Anti-memorization control (PLANNED, the rigorous follow-on)
This smoke uses a **RANDOM** holdout, which does NOT defend against memorization: an LLM
can recall a held-out (drug, disease) indication it saw in training. The rigorous study
is a **TEMPORAL holdout** — hold out indications for drugs approved (or DrugBank entries
added) after a cutoff the LLM's training likely predates, so knowledge-only cannot
recall them and any structure lift is real. Noted here as the decisive next step; the
memorization gauge below tells us whether it is necessary (it is).

## Token budget (smoke)
- Free/non-LLM baseline: runs fully (all seeds × k).
- Qwen (`qwen2.5:14b-instruct`, clpc35, free): ALL 12 prompts (3 conditions × 2 seeds ×
  k∈{1,3}).
- **Opus: 6 calls total** (3 conditions × 2 seeds × k=3 only — the stronger few-shot
  setting; ≤8 budget). Each prompt <10K tokens (knowledge ≈0.5K, anon ≈3.0K, fusion
  ≈3.3K tokens). Total Opus ≈ 6 × ~4K in + ~0.1K out ≈ **~25K tokens** (well under 100K).
- SMOKE grid: SEEDS=[11,22], K_SHOTS=[1,3], NQ=12.

## Files
- `eval/fusion_kg.py` — generator: 3 conditions + non-LLM baseline; writes
  `bench_out/fusion_kg/hetionet/{knowledge,anon,fusion}/seed*_k*.txt` (run_opus_cli.py /
  run_qwen.py compatible) + `manifest.json` (truth + baseline). No LLM. Reuses
  `graphlex.facts`/`verbalize`; graphlex core NOT modified.
- `eval/score_fusion_kg.py` — balanced-acc (+AUC) scorer; per condition × model; emits
  the decisive read + memorization gauge.
- This note.

## RESULTS (SMOKE; balanced accuracy, chance 0.500; pooled over seeds×k as noted)

Non-LLM structural reference (leakage-stripped graph):
| baseline           | bal-acc        | AUC            |
|--------------------|----------------|----------------|
| structural-logreg  | 0.646 ± 0.091  | 0.660 ± 0.212  |

graphlex+LLM, balanced accuracy (Opus: k=3 only, n=2 seeds; Qwen: pooled k∈{1,3}, n=4):
| condition        | Opus (k=3)     | Qwen (pooled)  |
|------------------|----------------|----------------|
| knowledge-only   | **0.917**      | 0.875 ± 0.072  |
| anon-structure   | 0.750          | 0.667 ± 0.102  |
| fusion           | 0.917          | 0.833 ± 0.059  |

## THE DECISIVE READ
**fusion vs max(knowledge-only, anon-structure):**
- Opus:  fusion 0.917 − max(0.917, 0.750) = **+0.000** → fusion does NOT clear both.
- Qwen:  fusion 0.833 − max(0.875, 0.667) = **−0.042** → fusion does NOT clear both.

For BOTH models fusion ties or trails the knowledge-only arm; the structure signal
(anon 0.67–0.75, above the 0.65 classical baseline and above 0.5 chance — so topology IS
informative) does not add on top of names. **The killer-app hypothesis is NOT supported
by this smoke.**

**Memorization gauge (knowledge-only absolute level):** Opus 0.917, Qwen 0.875 — both
NEAR-CEILING from NAMES ALONE, with no graph at all. This is the dominant finding: on a
random holdout the model recalls Hetionet/DrugBank indications parametrically, which
both (a) explains why fusion can't beat it (there is little headroom for structure to
add) and (b) flags that the random-holdout result is confounded by memorization.

## Verdict / next step
- Fusion signal: **NOT present** in this smoke (fusion ≤ knowledge-only for both
  models). Reported straight — this is a real, useful negative.
- Structure IS weakly informative (anon > chance and > classical baseline), so the
  *capability* exists; it is simply swamped by near-perfect parametric recall on a
  random holdout.
- BEFORE investing in the full study, run the **TEMPORAL-HOLDOUT** version: the random
  holdout is memorization-confounded (gauge ≈0.9), so the current knowledge-only ceiling
  is not trustworthy. The fusion-vs-both question is only meaningful once knowledge-only
  is knocked off the ceiling by held-out-in-time indications. If, under a temporal
  holdout, knowledge-only drops and fusion then exceeds both anon and knowledge, the
  killer-app claim revives. If knowledge-only stays high even temporally, the task is
  memorized and this is not the experiment to take to Nature.
- Recommendation: do the temporal-holdout study as the gate; do NOT scale the
  random-holdout version.
