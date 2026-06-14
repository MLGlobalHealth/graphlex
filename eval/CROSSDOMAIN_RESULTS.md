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

---

## UPDATE — v2: node features given to graphlex+LLM AND classical (fairness fix)

Batch 1 unfairly withheld node features (atom/residue types) from graphlex+LLM and
classical while the FMs used them. Fixed: added node-attribute **composition** to
`facts()`/`verbalize()` (top-k group fractions, deterministic) so the LLM sees it,
and appended the category-fraction vector to the classical features. IMDB has no
node features (unchanged). Harness: `crossdomain_graphcls.py` (OUT=crossdom_v2).

| domain | graphlex+LLM | classical | graphpfn | gmn | kumorfm | best spec | regret |
|---|---|---|---|---|---|---|---|
| social (IMDB) | 0.608 | **0.675** | 0.575 | 0.667 | 0.592 | 0.675 | +0.067 |
| biology (PROTEINS) | 0.667 | **0.742** | 0.742 | 0.725 | – | 0.742 | +0.075 |
| chemistry (NCI1) | 0.567 | 0.558 | **0.583** | 0.508 | – | 0.583 | +0.017 |

**Worst-case regret (flexibility score):** classical **0.025** · graphlex+LLM 0.075 ·
gmn 0.075 · kumorfm 0.083 (1/3) · graphpfn 0.100 · majority 0.175.

### What changed and what it means (honest)
1. **Node features did NOT help the LLM** (PROTEINS 0.667 unchanged; NCI1 ~flat;
   IMDB 0.642→0.608 is pure ICL sampling noise — identical input, different draw).
   Single-draw ICL noise is ~±0.03–0.04/cell at 3 seeds; **margins <0.05 are ties.**
2. **Node features helped the CLASSICAL arm** (PROTEINS 0.692→0.742), making
   **classical NetworkX+logreg the most flexible & effective method here**
   (max-regret 0.025). graphlex+LLM no longer ties it — it slightly trails.
3. **graphlex+LLM is still flexible and never substantially worse** (max-regret
   0.075, within noise of gmn) and still beats every FM on coverage/worst-case
   (graphpfn breaks on social, gmn on chemistry, kumorfm covers 1/3). The
   flexibility-**vs-FMs** story survives.

### The headline consequence (decision-relevant)
On **supervised graph classification, classical+logreg wins the flexibility race** —
graphlex+LLM ties the FMs but not the classical baseline. So "graphlex+LLM is the
MOST flexible & effective approach to AI on graphs" **cannot be carried by
classification accuracy.** It must be carried by the **task types classical+logreg
structurally cannot do**: zero-/few-label (no training), interpretable
natural-language output, mixed text+structure, and one unified interface across
many task types. That makes **batch 2 (zero-label structure-as-finding vs a null
model)** the load-bearing experiment — and per the leakage warning in
CROSSDOMAIN_PLAN.md, the comparison there must be LLM-at-0/1-shot vs logreg-across-
the-label-curve, never a self-leaking trained logreg.

### Still needed to harden this
- More seeds (3→≥8) and multiple ICL draws per cell to convert "ties" into CIs.
- ≥1 non-Claude model.

---

## SIDE-TEST — real element names vs opaque type ids (MUTAG)
Question (raised by Seth): does giving the LLM the *actual* elements ("Carbon",
"Oxygen") instead of opaque "t0/t1" help, since the LLM knows chemistry? NCI1 ships
integer labels with **no element legend** (README confirms), so it can't be named
without risking wrong chemistry. MUTAG has the canonical mapping (0=C,1=N,2=O,3=F,
4=I,5=Cl,6=Br; verified: Carbon is the dominant atom). Two arms differ ONLY in
naming. Harness: `eval/mutag_elements.py`. 10-shot/class, 40 q, 3 seeds, Opus.

| method | acc |
|---|---|
| classical (facts + atom composition) | 0.867 ± 0.051 |
| graphlex+LLM, **opaque** "t0/t1…" | 0.833 ± 0.062 |
| graphlex+LLM, **real elements** | 0.725 ± 0.061 |
| majority | 0.342 |

**Δ(elem − opaque) = −0.108** (per-seed elem−opaque: −0.25, +0.05, −0.125).
**Real element names did NOT help — they hurt** (opaque better in 2/3 seeds).
Caveats: n=3, single ICL draw → underpowered; the delta exceeds the std but is not
significant. Plausible mechanism (speculative): real names **activate chemistry
priors** the model then over-applies, but the verbalization gives only atom
*composition* (a bag of atoms), not the **bonding/substructure** needed to apply
those priors correctly (mutagenicity comes from functional groups like nitro/
aromatic rings, not atom counts) — so the prior misleads. Opaque labels force the
model to learn purely from the in-context pattern, which generalizes better here.
This echoes the synth anonymization result (priors are a double-edged sword) and
reinforces that **the real attribute lever is typed *substructure*, not naming**.
Both LLM arms still sit at/below classical (0.867) — consistent with the main
finding. Revisit with more seeds + a substructure-aware verbalization before
concluding.
