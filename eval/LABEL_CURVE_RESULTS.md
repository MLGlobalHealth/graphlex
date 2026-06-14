# Label-efficiency crossover — is the LLM "strictly worse than logreg if it can run"? (2026-06-14)

Motivation: across the 12-shot classification batches, logreg tied-or-beat
graphlex+LLM, suggesting "LLM is strictly worse than logreg whenever logreg can be
trained." This experiment tests that at the SAME *low* label budgets (k≥1, where
logreg runs but with little data). Harness: `eval/label_curve.py`. Opus, pure-ICL
subagents, 3 seeds, 30 queries. LLM gets k labeled examples/class in context;
logreg trains on the identical k. Domains: family (strong network-science prior)
and PROTEINS (real bio; CLASS0/1 are arbitrary to the model — weak prior).

## Result: NO — in the low-label regime the LLM is MORE label-efficient

### FAMILY (chance 0.333)
| k / class | graphlex+LLM | logreg | winner |
|---|---|---|---|
| 1 | 0.900 | 0.644 | **LLM +0.256** |
| 2 | – | 0.711 | |
| 3 | 0.889 | 0.822 | LLM +0.067 |
| 5 | 0.900 | 0.900 | tie |
| 8 | – | 0.900 | |
| 12 | – | 0.922 | logreg |

### PROTEINS (chance 0.500)
| k / class | graphlex+LLM | logreg | winner |
|---|---|---|---|
| 1 | 0.489 | 0.456 | LLM +0.033 |
| 2 | – | 0.522 | |
| 3 | 0.644 | 0.589 | LLM +0.055 |
| 5 | 0.611 | 0.589 | LLM +0.022 |
| 8 | – | 0.644 | logreg |
| 12 | – | 0.611 | |

## Reading
- **The "strictly worse" claim was an artifact of the 12-shot budget** (past the
  crossover). At k ≤ 5 the LLM beats logreg on **both** domains; logreg overtakes
  only with more labels (crossover ~k=5 family, ~k=8 PROTEINS).
- **The LLM's prior substitutes for training data** — it starts high and is flat in
  k; logreg starts near chance and climbs. The gap is largest at k=1 (family LLM
  0.900 vs logreg 0.644) and shrinks as labels accumulate.
- **It holds even without a strong domain prior** (PROTEINS, arbitrary classes):
  the LLM is still ≥ logreg at k=1,3,5 — smaller margins, but the *direction* is
  consistent. So label-efficiency isn't only a network-science-prior effect.

## Why this matters for the PNAS "effective" claim
This converts the story from "LLM ties classical at best" to a concrete
**effectiveness** win: **graphlex+LLM is the most label-efficient option — it
dominates the small-label regime where small-network science actually operates, and
loses only when labels are plentiful** (where you'd train a model anyway, and where
classical+logreg remains the right tool). Combined with the zero-label result
(`ZEROLABEL_RESULTS.md`), the regime map is:

| label budget | best option |
|---|---|
| 0 labels | graphlex+LLM (only thing that runs) |
| 1–~5 / class | graphlex+LLM (more label-efficient) |
| many / class | classical+logreg (cheaper, ties-or-better) |

## Caveats
- 3 seeds, single ICL draw — family margins at k=1,3 are large/robust; PROTEINS
  margins (+0.02–0.06) are small and near chance, within noise. Direction is
  consistent but PROTEINS needs ≥8 seeds + multi-draw to be significant.
- Crossover location is approximate (coarse k grid; LLM tested at 1,3,5 only).
- Single LLM family (Claude/Opus); replicate on a non-Claude model.
- family is synthetic (clean ground truth); the prior-driven head-start there is
  partly the model knowing ER/BA/WS — which is the point (priors = free labels),
  but means the family gap is an upper bound on the effect.

---

## HARDENED (2026-06-14): 8 seeds, 3 domains, TWO model families

Added IMDB (social/weak-prior), bumped to 8 seeds, and added a non-Claude model:
**Qwen2.5-14B-Instruct** run locally on clpc35 (Ollama; driver `eval/run_qwen.py`,
temp 0). Opus anchored at 3 seeds; Qwen + logreg at 8 seeds. **This revises the
3-seed conclusion above.**

### FAMILY (chance 0.333) — mean ± std
| k | Opus (3s) | Qwen-14B (8s) | logreg (8s) |
|---|---|---|---|
| 1 | **0.878 ± 0.079** | 0.521 ± 0.120 | 0.567 ± 0.174 |
| 3 | **0.900 ± 0.047** | 0.658 ± 0.089 | 0.804 ± 0.098 |
| 5 | 0.922 ± 0.016 | 0.708 ± 0.062 | 0.887 ± 0.104 |

### PROTEINS (chance 0.500) — mean ± std
| k | Opus (3s) | Qwen-14B (8s) | logreg (8s) |
|---|---|---|---|
| 1 | 0.489 ± 0.042 | 0.567 ± 0.131 | 0.533 ± 0.093 |
| 3 | 0.622 ± 0.063 | 0.604 ± 0.111 | 0.621 ± 0.074 |
| 5 | 0.522 ± 0.042 | 0.587 ± 0.100 | 0.621 ± 0.078 |

### IMDB (chance 0.500) — mean ± std
| k | Opus (3s) | Qwen-14B (8s) | logreg (8s) |
|---|---|---|---|
| 1 | 0.433 ± 0.000 | 0.471 ± 0.106 | 0.475 ± 0.081 |
| 3 | 0.489 ± 0.096 | 0.575 ± 0.115 | 0.529 ± 0.096 |
| 5 | 0.444 ± 0.150 | 0.517 ± 0.104 | 0.517 ± 0.107 |

> Correction (2026-06-14): the Qwen-14B PROTEINS/IMDB cells above were re-scored after
> a parser bug fix — `score_labelcurve.py` had used a strict regex that silently
> dropped 9 Qwen answer files in "Query N CLASS" format, so those cells had been
> computed on 3–7 seeds instead of 8 (PROTEINS k=3 0.578→0.604, k=5 0.633→0.587;
> IMDB k=5 0.514→0.517). Family and all Opus numbers were unaffected (plain R/S/W
> format parses identically); the label-efficiency figure already used the tolerant
> parser. Conclusions unchanged (PROTEINS/IMDB remain near-chance for all methods).

### Revised, honest conclusion
1. **The low-label LLM-beats-logreg win is real but NARROW: it requires a capable
   model AND a prior-rich domain.** The only robust instance is **Opus on family**
   (k=1: 0.878 vs logreg 0.567, ±0.08 — large and stable; k=3: 0.900 vs 0.804).
2. **It is NOT a generic "LLM" property.** **Qwen-14B does not beat logreg** — it is
   *below* logreg on family at k=3,5 (0.658/0.708 vs 0.804/0.887). A weaker model
   loses the advantage entirely.
3. **On weak-prior real graphs (PROTEINS, IMDB) it's a wash near chance** — Opus,
   Qwen, and logreg all sit within noise of each other (and of chance). No LLM
   advantage, but not "strictly worse" either. (Structure-only is just weak here.)
4. So the earlier 3-seed "LLM ≥ logreg at low k on both domains" was **over-read**;
   with 8 seeds + a second model the PROTEINS edge vanishes into noise and the
   family win turns out to be Opus-specific (capability-gated).

**Implication for the paper:** the "effective" (beats logreg) claim must be scoped
to *frontier-model + prior-rich task* and stated with the model-capability
dependence shown explicitly. The broad, safe claim remains the **zero-label
capability** (logreg can't run at all) + flexibility; the *beats-logreg-with-labels*
claim is a narrower, capability-gated result.

### Capability ladder — Qwen-32B added (it IS model scale, not Claude-specific)
Qwen2.5-32B-Instruct-q4 (8 seeds, clpc35) on family:

| k | Opus | Qwen-32B | logreg | Qwen-14B |
|---|---|---|---|---|
| 1 | **0.878** | **0.662** | 0.567 | 0.521 |
| 3 | 0.900 | 0.729 | 0.804 | 0.658 |
| 5 | 0.922 | 0.792 | 0.887 | 0.708 |

- **The low-label win scales monotonically with model capability.** At k=1:
  Opus (0.878) ≫ Qwen-32B (0.662) > logreg (0.567) > Qwen-14B (0.521). Scaling
  14B→32B *recovers* a k=1 win over logreg that 14B lacked → the effect is **model
  capability, NOT Opus-specific.**
- **But for non-frontier models the win is fragile:** Qwen-32B beats logreg only at
  k=1; by k=3,5 logreg overtakes it (0.804/0.887 vs 0.729/0.792). Only the frontier
  model (Opus) stays above logreg across k=1–5.

**Refined paper claim (clean + honest):** *On prior-rich tasks in the low-label
regime, graphlex+LLM beats trained logreg, and the margin scales with model
capability — frontier models win across the low-label regime; a mid-size open model
(32B) wins only at the extreme low end (k=1); a small model (14B) does not.* This
is a capability-axis result (an existence ladder), not a single-model anecdote.
Still domain-gated: holds on family (network-science prior), washes out near chance
on structure-only PROTEINS/IMDB.
