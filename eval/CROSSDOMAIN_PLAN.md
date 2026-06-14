# Cross-domain experiment program — the "flexibility" headline (2026-06-14)

**Target claim (PNAS):** *graphlex + a frozen LLM is the most flexible and
effective general approach to AI on graphs — a single training-free, interpretable
pipeline that is competitive across scientific domains and task types, while every
specialized method is strong only on its home turf.*

The empirical bar is **not** "win everywhere." It is:
> **(F1) graphlex+LLM is never *substantially worse* than the best specialist in
> any (domain × task) cell, AND (F2) no specialist satisfies F1 across all cells.**

That asymmetry — low *worst-case regret* for one general method vs high worst-case
regret for every specialist — is the headline figure.

---

## The design: a (domain × task-type) matrix

### Scientific domains (rows) — breadth is the point
1. **Chemistry / molecules** — MUTAG, NCI1, (BBBP/BACE if needed). Node attrs = atom types.
2. **Structural biology** — PROTEINS, ENZYMES, DD. Node attrs = residue/SSE types.
3. **Social / collaboration** — IMDB-BINARY, COLLAB, REDDIT-BINARY. Often featureless.
4. **Information / web** — a non-citation web/knowledge graph (leakage-safe).
5. **Neuroscience / connectomics** — connectome subgraphs (if data accessible).
6. **Ecology / infrastructure** — food webs / power grid / transport (structure-only).

First pass covers 1–3 (data in hand); 4–6 are the breadth extension that pushes
toward PNAS-main.

### Task types (columns) — each is a different "AI on graphs" capability
- **A. Graph classification** (whole-graph property) — chem/bio/social.
- **B. Node classification** (attribute from position+neighbors) — incl. the
  de-confounded relational task (`fair_node_hard`).
- **C. Link prediction** (does an edge exist) — common-neighbors / Adamic-Adar regime.
- **D. Structure-as-finding / model selection vs null** (is it small-world /
  scale-free / modular beyond a configuration-model null) — graphlex's native mode;
  the synth family-ID result is the prototype.
- **E. Graph-level regression / property estimation** (optional, later).

### The "specialists" each cell is scored against
Per cell, take the **best** of the relevant baselines as the specialist bar:
- **Trained GNN** (GIN for graph-cls, GCN/SGC for node) — the deep-learning specialist.
- **logreg on classical NetworkX features** — the classical specialist (often SOTA
  on small graphs; see project memory).
- **untrained random GNN** — control (frequently matches trained FMs).
- **task-trivial** — majority class (A), neighbor-majority (B), common-neighbors (C).
- **KumoRFM** where the task is genuinely relational/tabular (B), with the *correct*
  schema (open item; ask Kumo) — never report the misconfigured native-mode number.

**graphlex+LLM** = one pipeline for every cell: `facts()`→`verbalize()` (+ node-attr
verbalization where present) → frozen-LLM in-context learning. No training, no
per-task tuning beyond the shots in context.

---

## Metrics
- **Per cell:** accuracy/AUC mean ± CI over ≥5 seeds; n_query ≥ ~40/seed.
- **Regret(method, cell) = best_specialist(cell) − method(cell).** The core quantity.
- **Worst-case regret** per method across all cells (the flexibility score).
- **F1 threshold ("substantially worse"):** pre-register δ. Proposal: a method
  "matches" a cell if regret ≤ max(0.05, the specialist's 95% CI half-width). Tune
  after pilot; report the raw regret heatmap regardless so the claim isn't
  threshold-dependent.

## Headline display items
- **F1 (the money figure):** regret heatmap, rows = methods, cols = (domain×task)
  cells. graphlex+LLM row ≈ all-green (low regret); every specialist row has red cells.
- **F2:** per-domain bars, graphlex+LLM vs best specialist, with CIs.
- **F3:** the permutation/anonymization controls (from the pilot) as the mechanism
  panel — *why* the verbalized approach generalizes and isn't memorization.
- **T1:** full numbers, all cells × methods × seeds.

---

## Honesty guards (carry from PILOT_RESULTS.md)
- graphlex+LLM **matches** classical/GNN; it does not generally beat them. The claim
  is *flexibility + competitiveness*, not raw SOTA. Keep that wording exact.
- Multi-model (≥1 non-Claude) needed before "LLMs", not "Claude".
- Leakage control on any real network the LLM might have memorized.
- Same verbalization protocol across all cells (uniformity = the flexibility proof);
  no per-cell prompt engineering beyond declared shot count.
- Pure-ICL subagents (verify ≤2 tool uses: Read+Write, no computation).

---

## Execution order
1. **Batch 1 (now):** Task A (graph classification) across chem/bio/social
   (MUTAG, PROTEINS or ENZYMES, IMDB-BINARY) — graphlex-verbalize+Claude vs
   logreg+GIN+majority. Proves the harness, first cross-domain regret numbers.
2. Batch 2: Task D (structure-vs-null) across domains — graphlex's native strength.
   **LEAKAGE WARNING (verified in Jess's baseline.py + stats.py):** for
   structure-as-finding / structural-property targets, logreg-on-structural-features
   reads the answer off its own input — `NetworkStatsEncoder` is documented to score
   R²=1.000 on exactly these probe targets, and synth family-ID logreg already hit
   0.92 for the same reason. So in batch 2 a trained logreg is NOT a weak baseline
   and "logreg can't do it" is FALSE. graphlex+LLM's honest, *capability* edge here
   is **zero labels** (logreg must be trained; the verbalize→null-model→LLM pipeline
   needs none) + interpretable null-model reasoning — not a margin win. Design batch 2
   as: LLM at 0/1-shot vs logreg across the label curve; never present a
   self-leaking trained logreg as something the LLM "beats".
3. Batch 3: Tasks B & C (node / link) across domains.
4. Add domains 4–6; add a non-Claude model; leakage controls.
5. Assemble the regret heatmap; decide PNAS-main readiness.
