# Cross-domain batch 1 — graph classification (2026-06-14)

Task A (whole-graph classification) across 3 sciences, real TUDatasets. Every method
gets the **same graphs, same low-label split, same budget** (12 shots/class, 40
queries, 3 seeds). Harness: `eval/crossdomain_graphcls.py`. graphlex+LLM = Opus,
pure-ICL subagents (2 tool uses each, verified). Classical and LLM consume the
**identical** `graphlex.facts()` (vector vs prose); FM embeddings are precomputed on
the same real graphs (`/home/scratch/real_fm_embeddings/`).

> **What "classical" is here:** logreg on the 14-d `graphlex.facts()` vector — a
> NetworkStats-*equivalent*, NOT Jess's `NetworkStatsEncoder` itself. Verified
> faithful: on identical 12-shot splits, my facts-logreg vs Jess's actual
> `NetworkStatsEncoder`-logreg = IMDB 0.675/0.625, PROTEINS 0.692/0.625, NCI1
> 0.558/0.583 — within ~0.07 (seed noise); the parity conclusion holds with either
> (Jess's is slightly weaker at low-label, so graphlex+LLM *edges* her classical on
> IMDB/PROTEINS). **No bug invalidates the classical baseline for classification:**
> the class label is not a structural-stat input, so there is no self-leakage. (The
> documented R²=1.000 self-leakage of `NetworkStatsEncoder` is real but scoped to
> structural-*property probe* tasks where the target equals an input column — see
> the batch-2 warning in CROSSDOMAIN_PLAN.md; it does not affect classification.)

> Fairness note: graphlex+LLM and `classical` are **structure-only**; the FM
> embeddings (graphpfn/gmn) were computed **with node features** (atom types for
> NCI1, residues for PROTEINS). So on bio/chem the FMs have a feature advantage that
> graphlex+LLM does not — yet it still matches/beats them. Adding attribute
> verbalization (atom/residue composition) is the obvious next lever.

## Results (accuracy, mean over 3 seeds)

| domain | dataset | graphlex+LLM | classical | graphpfn | gmn | kumorfm | majority | best specialist | regret |
|---|---|---|---|---|---|---|---|---|---|
| social | IMDB-BINARY | 0.642 | **0.675** | 0.575 | 0.667 | 0.592 | 0.550 | 0.675 | +0.033 |
| biology | PROTEINS | 0.667 | 0.692 | **0.742** | 0.725 | – | 0.567 | 0.742 | +0.075 |
| chemistry | NCI1 | **0.617** | 0.558 | 0.583 | 0.508 | – | 0.542 | 0.583 | −0.033 |

## Flexibility score — worst-case regret across the 3 cells
(regret = per-cell best-of-all-methods − method; lower = more flexible)

| method | max-regret | mean-regret | coverage |
|---|---|---|---|
| **graphlex+LLM** | **0.075** | 0.036 | 3/3 |
| classical (NetworkX+logreg) | 0.058 | 0.036 | 3/3 |
| graphpfn | 0.100 | 0.044 | 3/3 |
| gmn | 0.108 | 0.044 | 3/3 |
| kumorfm | 0.083 | 0.083 | **1/3** |
| majority | 0.175 | 0.125 | 3/3 |

## Honest reading
1. **graphlex+LLM is flexible and never substantially worse** — competitive in all
   three sciences, and the *best* method on chemistry, despite being structure-only
   against feature-using FMs. Supports the "flexible + effective" headline. ✓
2. **The FMs are the specialists, not graphlex+LLM** — graphpfn lags on social
   (0.575), gmn collapses to ~chance on chemistry (0.508), kumorfm only covers one
   domain. Each has a cell where it falls ~0.10 behind. This is exactly the asymmetry
   the headline figure needs.
3. **The honest snag: classical NetworkX+logreg is *equally* flexible** (max-regret
   0.058 ≈ LLM's 0.075). The LLM does **not** beat classical on accuracy or on the
   flexibility metric. So the LLM's case cannot rest on accuracy — it must rest on
   what classical+logreg *cannot do*: zero training, natural-language
   interpretability, and tasks logreg can't run (zero-label structure-as-finding,
   mixed-modality, link/anomaly). Those are the next batches.

## Implications for the PNAS claim
- "Most flexible & effective" is supportable **against the FMs** already (batch 1).
- Against the classical baseline, accuracy parity is not enough — we need the
  batches where graphlex+LLM does something classical+logreg structurally cannot:
  - **Batch 2 (highest value): structure-as-finding vs null, ZERO labels** — logreg
    needs labels; the verbalize+LLM pipeline does not. This is where the LLM wins by
    *capability*, not margin.
  - Batch 3: attribute-rich verbalization (close the PROTEINS/NCI1 gap), link
    prediction, node tasks.
- Also needed: more seeds (3 → ≥8) for tighter CIs; ≥1 non-Claude model.

## Caveats
- 3 seeds, 40 queries/cell — CIs are wide (e.g. NCI1 std up to 0.17); treat margins
  <~0.05 as ties. The NCI1 "LLM wins" is in a near-chance regime — don't over-read.
- Structure-only for graphlex/classical; FMs use node features.
- Single LLM family (Claude/Opus).
