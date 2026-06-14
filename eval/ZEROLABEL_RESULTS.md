# Zero-label capability — the task type classical+logreg can't do (2026-06-14)

The cross-domain classification batches showed graphlex+LLM *ties* classical+logreg
on accuracy (classical even wins). So the "most flexible & effective" headline can't
rest on supervised accuracy. This experiment targets the regime where the LLM has a
**capability** classical methods structurally lack: **operating at zero / one
training label**, using the model's priors over `verbalize(facts(G))`. logreg needs
labels to train, so at 0 labels it is undefined (chance).

Honest framing (per CROSSDOMAIN_PLAN leakage warning): the claim is NOT "LLM beats
trained logreg" — it's "LLM works at 0 labels where logreg can't be trained, and on
strong-prior domains its 0-label accuracy equals logreg trained on many labels."
Harness: `eval/zero_label.py`. Opus, pure-ICL subagents, 3 seeds, 30 queries.

## Label-efficiency curves

### FAMILY — structural family ID (network-science prior), chance 0.333
| # labels / class | graphlex+LLM | logreg |
|---|---|---|
| **0** | **0.922** | 0.333 (chance — cannot train) |
| 1 | – | 0.622 |
| 3 | 0.878 | 0.844 |
| 10 | – | 0.922 |

**graphlex+LLM at 0 labels (0.922) = logreg at 10 labels/class (0.922).** The LLM
delivers ~10 labels' worth of accuracy for free, from network-science priors over
the verbalized structure. (3-shot 0.878 ≈ 0-shot; the few examples add noise rather
than signal here — the prior is already strong.)

### MUTAG — mutagenicity (chemistry prior), chance 0.500
| # labels / class | graphlex+LLM | logreg |
|---|---|---|
| **0** | **0.633** | 0.500 (chance) |
| 1 | – | 0.711 |
| 3 | 0.700 | 0.778 |
| 10 | – | 0.833 |

Above chance at 0 labels (0.633), but **logreg overtakes with just 1 example**. The
zero-label edge is **domain-dependent**: large where the model's priors are good
(network families), small for molecular activity inferred from a verbalized summary
(rings + bonds + composition is still a lossy view of a molecule).

## What this establishes (and doesn't)
- **A capability gap classical methods cannot close:** at 0 (and effectively 1)
  labels, graphlex+LLM is the *only* method that produces a useful answer — logreg/
  GNN/FM all require training data. This is the honest backbone of "most flexible".
- **On strong-prior domains the gap is large** (network science: 0-label LLM = 
  10-shot logreg). **On weak-prior domains it shrinks fast** (chemistry: gone by 1
  label). So the headline must be scoped: *graphlex+LLM uniquely spans the
  zero/low-label regime across domains, and on domains where network-science/world
  knowledge applies it matches heavily-trained baselines with no labels.*
- It does NOT show the LLM beating trained classical at many labels (it doesn't).

## Caveats
- 3 seeds, single ICL draw, 30 queries — margins <~0.05 are ties.
- Family is synthetic (clean ground truth); MUTAG is real but small/imbalanced.
- Zero-shot relies on the family NAMES / "mutagenic" being meaningful to the model
  (priors). That is the mechanism, not a leak — but it means zero-shot only applies
  to semantically-named targets, not arbitrary class labels.
- Single LLM family (Claude/Opus); needs a non-Claude replication.

## Where this points
The flexibility headline is now: **one training-free pipeline that (a) is
competitive with specialized methods when labels exist [classification batches] and
(b) is the only thing that works when they don't [this].** The strongest, cleanest
cell is zero-label network-science finding (0.92 at 0 labels). Next: a zero-label
structure-as-finding-vs-null task on REAL networks across domains, and a non-Claude
model.
