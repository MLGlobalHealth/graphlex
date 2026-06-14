# Broad cross-domain sweep — 25 real datasets, 6 sciences (2026-06-14)

Tests the flexibility headline at scale: is graphlex+LLM "competitive, never
substantially worse" across many real datasets from many scientific domains, at a
low-label budget? Harness: `eval/sweep.py` (+ `probe_datasets.py`, `run_qwen.py`,
`score_sweep.py`, `balanced_rescore.py`). 5 shots/class, 40 queries, structure +
node-attribute composition verbalization. graphlex+LLM = Opus (seed 11 anchor) and
Qwen2.5-14B (3 seeds, local on clpc35); baselines = classical (logreg on graphlex
facts) and majority. **All pure-ICL; logreg trained on the same shots.**

Domains (25 datasets): **chemistry** (MUTAG, NCI1, Mutagenicity, AIDS, BZR, COX2,
DHFR, PTC_MR), **biology** (PROTEINS, ENZYMES), **neuroscience/brain connectomes**
(KKI, OHSU, Peking_1), **social** (IMDB-B, IMDB-M, COLLAB, deezer, github_stargazers,
twitch_egos), **vision** (MSRC_21, Letter-high, Fingerprint), **synthetic** (Synthie,
COLORS-3, TRIANGLES). (DD, REDDIT-B skipped — graphs too large for the budget.)

## The metric matters: raw accuracy is unfair here
The LLM sees **balanced** 5/5 shots, so it has **no base-rate signal**; many test
sets are **imbalanced**. Raw accuracy then rewards "always predict the majority
class" — which the majority baseline does and the LLM can't. Under raw accuracy the
LLM looks bad on imbalanced sets (e.g. Opus on BZR: 0.375 vs majority 0.842).
**Balanced accuracy (macro-averaged per-class recall; majority = chance) is the fair
metric** and is what we report as primary.

## Balanced accuracy (primary) — majority == chance

| domain | dataset | chance | classical | Qwen-14B | Opus |
|---|---|---|---|---|---|
| chemistry | AIDS | 0.500 | 0.850 | 0.853 | **1.000** |
| chemistry | MUTAG | 0.500 | 0.760 | 0.616 | **0.833** |
| chemistry | COX2 | 0.500 | 0.561 | 0.598 | **0.688** |
| chemistry | Mutagenicity | 0.500 | 0.557 | **0.647** | 0.570 |
| chemistry | NCI1 | 0.500 | 0.506 | 0.566 | 0.372 |
| chemistry | BZR | 0.500 | 0.548 | 0.527 | 0.495 |
| chemistry | DHFR | 0.500 | 0.487 | 0.517 | 0.431 |
| chemistry | PTC_MR | 0.500 | 0.481 | 0.520 | 0.575 |
| biology | PROTEINS | 0.500 | 0.520 | 0.621 | **0.736** |
| biology | ENZYMES | 0.167 | 0.252 | – | 0.253 |
| neuroscience | OHSU | 0.500 | 0.520 | – | 0.544 |
| neuroscience | Peking_1 | 0.500 | 0.546 | – | 0.467 |
| neuroscience | KKI | 0.500 | 0.475 | – | 0.444 |
| social | twitch_egos | 0.500 | 0.660 | 0.666 | **0.735** |
| social | COLLAB | 0.333 | 0.607 | **0.683** | 0.628 |
| social | deezer_ego_nets | 0.500 | 0.560 | 0.601 | **0.681** |
| social | IMDB-MULTI | 0.333 | 0.381 | **0.492** | 0.336 |
| social | IMDB-BINARY | 0.500 | 0.526 | 0.517 | 0.518 |
| social | github_stargazers | 0.500 | 0.533 | 0.495 | 0.464 |
| vision | MSRC_21 | 0.050 | 0.730 | – | **0.882** |
| vision | Letter-high | 0.067 | 0.290 | 0.239 | 0.256 |
| vision | Fingerprint | 0.067 | 0.226 | 0.095 | 0.047 |
| synthetic | TRIANGLES | 0.100 | 0.674 | 0.653 | **1.000** |
| synthetic | Synthie | 0.250 | 0.480 | 0.486 | 0.509 |
| synthetic | COLORS-3 | 0.091 | 0.084 | 0.091 | 0.062 |

### Flexibility summary (balanced acc, vs best non-LLM baseline)
| model | mean regret | worst | within 0.05 | beats classical | subst. worse (>0.10) |
|---|---|---|---|---|---|
| **Opus (seed 11)** | **−0.026** | +0.179 | 18/25 | 14/25 | **2/25** (Fingerprint, NCI1) |
| **Qwen-14B (3 seeds)** | **−0.007** | +0.144 | 17/20 | 13/20 | **2/20** |

## What this establishes
1. **Under the fair metric, graphlex+LLM is competitive across all 6 sciences and
   never substantially worse — except 2/25 for Opus.** On average it is *slightly
   ahead* of the best non-LLM baseline (mean regret −0.026), with clear wins:
   AIDS 1.00, TRIANGLES 1.00, MSRC_21 0.88 (vs classical 0.73), MUTAG 0.83,
   PROTEINS 0.74, twitch 0.74, COX2 0.69, deezer 0.68.
2. **Even a small open model (Qwen-14B) matches the classical baseline on average**
   (mean regret −0.007) — so the broad flexibility is not Opus-specific (though the
   *margins* and the no-substantial-loss property are stronger for the frontier
   model, consistent with the capability ladder in LABEL_CURVE_RESULTS.md).
3. **The raw-accuracy "failures" were a base-rate artifact**, not LLM weakness: BZR
   regret collapses from +0.467 (raw) to +0.053 (balanced); COX2 flips from a loss
   to an Opus *win* (0.688 vs 0.561).
4. **TRIANGLES = 1.000 (Opus)** is a clean structural-reasoning win — the LLM counts
   triangles from the verbalized facts perfectly; classical logreg gets 0.674.

## Honest caveats
- **Opus is 1 seed** (seed 11); Qwen is 3 seeds. Opus needs multi-seed CIs before
  the per-dataset numbers are trustworthy (esp. small neuroscience sets, n≈80).
- **Genuine LLM weak spots:** Fingerprint (vision, near chance) and NCI1 (chem) for
  Opus; Fingerprint + github for Qwen.
- **5 datasets unscored for Qwen-14B** (ENZYMES, KKI, OHSU, Peking_1, MSRC_21): the
  14B model emitted prose instead of the answer format on long / many-class prompts.
  A format-robust prompt or the 32B model would recover these.
- Low-label (5/class); classical is also at 5/class. Different budgets would shift
  the picture (see LABEL_CURVE_RESULTS.md crossover).
- Several datasets are near chance for everyone (ENZYMES, COLORS-3, KKI) — weak
  signal, not a differentiator.

## EXPANDED (2026-06-14): 30 datasets across 8 sciences

Added 5 new datasets in genuinely new domains — **citation** (DBLP_v1), **archaeology**
(Cuneiform), plus FRANKENSTEIN (chem), MSRC_9 (vision), SYNTHETICnew (synthetic).
(Attempted robotics/FIRSTMM_DB but its graphs are too large + only 41 graphs for 11
classes → skipped; COIL-DEL 100-class and REDDIT-MULTI-5K also skipped.) DBLP's
41k-dim bag-of-words features exceed the composition cap → structure-only. Opus now
covers all 30 (seed 11; + seeds 22/33 on 6 domain reps). Balanced accuracy:

### FINAL — full 3-seed Opus on all 30 (the publishable aggregate)
| model | n | seeds | mean regret | worst | within 0.05 | beats classical | subst. worse (>0.10) |
|---|---|---|---|---|---|---|---|
| **Opus** | 30 | 3 | **−0.016** (ahead) | **+0.099** | **25/30** | 18/30 | **0/30** |
| Qwen-14B | 21 | 3 | −0.000 | +0.144 | 17/21 | 13/21 | 3/21 |
| Qwen-32B-q4 | 27 | 3 | +0.021 | +0.182 | 16/27 | 12/27 | 6/27 |

**With full multi-seed, Opus is NEVER substantially worse (0/30), ahead of the best
classical baseline on average (−0.016), within 0.05 on 25/30, and wins outright on
18/30 — across 8 sciences.** This is the clean, robust flexibility result.

> **Critical lesson: single-seed Opus was misleadingly pessimistic.** The "3
> failures" reported at 1 seed were ALL artifacts of which graphs landed in the tiny
> 5-shot set: DBLP_v1 0.343→**0.604**, NCI1 0.372→**0.569** (now *beats* classical),
> Fingerprint 0.047→**0.127**. Multi-seed didn't just add CIs — it overturned the
> conclusion. **Always report ≥3-seed.** (Qwen-32B-q4 ≈/below 14B — quantized scale
> is not a lever on arbitrary-label graph-cls, unlike the network-science capability
> ladder.)

### DBLP_v1 failure — diagnosed (it was the seed, not the model)
A dedicated diagnosis (workflow phase 2) found the seed11 below-chance result
(0.343) was **5-shot-set contamination**, not a model deficit. The separating
feature is clean: **CLASS1 = small dense cliques (mean density 0.81, ~70% perfect
cliques), CLASS0 = larger sparse graphs (density 0.33)** — logreg learns "clique ⇒
CLASS1" (0.87 train acc; top coefs avg_clustering +2.0, n_nodes −1.6). seed11's
shot set happened to include 2 CLASS0 *cliques* and a sparse CLASS1, contradicting
the heuristic → Opus inferred an anti-correlated rule and inverted 12/17 sparse
true-CLASS0 queries. seeds 22/33 (clean shot sets) score 0.75/0.72. Lesson: with
arbitrary integer class labels and a 5-shot budget, a contaminated demo set can flip
the LLM's boundary on one seed — exactly why multi-seed is non-negotiable here.

Note: the snapshot numbers below were 1-seed Opus for most datasets and are
SUPERSEDED by the full 3-seed FINAL aggregate at the end of this section. The
1-seed snapshot suggested "3 failures (DBLP/NCI1/Fingerprint)" — multi-seed showed
all three were seed artifacts (see FINAL + DBLP diagnosis). Qwen-32B-q4 ≈/below 14B
on these tasks; Qwen-32B fixed the 14B format failures (KKI/OHSU/ENZYMES) except
MSRC_21 (20-class, both Qwen sizes ramble).

## Takeaway for the paper
At low labels, across 25 datasets and 6 sciences, **one training-free pipeline
(graphlex+LLM) is competitive with — and on average slightly better than — a
trained classical baseline, and substantially worse on only ~2/25, under the fair
(balanced-accuracy) metric.** That is the concrete, multi-domain backing for "most
flexible & effective," with the honest scoping: frontier model for the
no-substantial-loss guarantee; metric must be balanced accuracy; multi-seed Opus
still needed.
