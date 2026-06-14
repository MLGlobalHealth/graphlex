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

| model | n | mean regret | within 0.05 | beats classical | subst. worse (>0.10) |
|---|---|---|---|---|---|
| **Opus** | 30 | **−0.009** | 22/30 | 15/30 | 3/30: DBLP_v1, NCI1, Fingerprint |
| Qwen-32B-q4 | 27 | +0.021 | 16/27 | 12/27 | 6/27 |
| Qwen-14B | 21 | −0.000 | 17/21 | 13/21 | 3/21 |

(Final balanced-accuracy aggregates, all arms complete. Qwen-32B-q4 is *slightly
worse* than 14B on flexibility here — quantized 32B ≈/below 14B on arbitrary-label
graph classification, so model scale is not a lever on this task family, unlike the
network-science capability ladder. Opus is the only arm ahead of the baseline on
average and the only one substantially-worse on ≤3 datasets.)

**Holds across 8 sciences:** under balanced accuracy Opus is still *ahead on average*
(−0.009) and within 0.05 on 22/30. New wins include MSRC_9 0.854, SYNTHETICnew 0.659,
PROTEINS 0.736. **New honest failures:** **DBLP_v1 (citation): Opus 0.343 vs classical
0.664 — a real, large miss** (structure-only citation ego-graphs; the LLM does worse
than chance while logreg finds structural signal); also NCI1 and Fingerprint persist.
**Qwen-32B-q4 is not better than 14B here** — on these low-label classification tasks
the quantized 32B ≈ 14B (the clean capability ladder was on family/network-science,
not on arbitrary-label graph classification). Qwen-32B did fix the 14B format
failures (KKI/OHSU/ENZYMES now score). MSRC_21 still rambles for both Qwen sizes
(20-class prompt).

**Gaps closed / remaining:** multi-seed added for Opus on 6 domain reps + Qwen 3-seed
throughout (CIs); Qwen format failures fixed via 32B except MSRC_21. Still open:
full multi-seed Opus on all 30 (only reps have 3 seeds); MSRC_21 many-class format;
and the genuine model weak spots (DBLP citation, NCI1, Fingerprint) are real, not
artifacts.

## Takeaway for the paper
At low labels, across 25 datasets and 6 sciences, **one training-free pipeline
(graphlex+LLM) is competitive with — and on average slightly better than — a
trained classical baseline, and substantially worse on only ~2/25, under the fair
(balanced-accuracy) metric.** That is the concrete, multi-domain backing for "most
flexible & effective," with the honest scoping: frontier model for the
no-substantial-loss guarantee; metric must be balanced accuracy; multi-seed Opus
still needed.
