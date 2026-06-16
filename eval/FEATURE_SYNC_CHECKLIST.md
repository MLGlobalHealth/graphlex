# Feature-coverage / 4-way sync checklist (lab note, 2026-06-15)

**The integrity condition for the headline comparison.** The paper's core claim is
"LLM reasoning over verbalized features vs. logreg over the *same* features." That is
only coherent if **four things carry the identical feature set**:

1. **canonical** — the full computed set in `src/graphfm/eval/properties.py` (the
   original NetworkStats "compute everything" vector); `/home/scratch/fmsn-dev`.
2. **facts()** — what `graphlex/core/facts.py` computes.
3. **verbalize()** — what the **LLM** actually sees (`graphlex/verbalize/render.py`).
4. **logreg X** — what the **classical 0-th-level baseline** is trained on
   (`eval/_common.py::fvec`, the 15-key `FKEYS`).

Today these DIVERGE. This note records the diff + the fix. Do NOT edit
facts()/verbalize()/fvec while experiments are running on the current verbalization
(it silently mixes versions). Implement after the in-flight runs finish; keep the
current run as the "impoverished/mismatched" before-ablation.

## The logreg vector today (`_common.fvec`, 15 dims = the "15-d vector")
n_nodes, n_edges, density, n_components, mean_degree, max_degree, degree_std,
max_over_mean_degree, avg_clustering, transitivity, degree_assortativity,
avg_path_length, diameter, n_cycles, n_communities

## Coverage table (✅ present, ❌ absent, ~ partial)

| feature | canonical | facts() | LLM (verbalize) | logreg (fvec) |
|---|---|---|---|---|
| density | ✅ | ✅ | ✅ | ✅ |
| mean_degree | ✅ | ✅ | ✅ | ✅ |
| degree_std | ✅ | ✅ | ✅ | ✅ |
| max_degree | – | ✅ | ✅ | ✅ |
| max/mean degree | – | ✅ | ✅ | ✅ |
| avg_clustering | ✅ | ✅ | ✅ | ✅ |
| transitivity | ✅ | ✅ | ✅ | ✅ |
| degree_assortativity | ✅ | ✅ | ✅ | ✅ |
| avg_path_length | ✅ | ✅ | ✅ | ✅ |
| diameter | ✅ | ✅ | ✅ | ✅ |
| n_components | – | ✅ | ✅ | ✅ |
| n_cycles | – | ✅ | ✅ | ✅ |
| n_communities | ✅ | ✅ | ✅ | ✅ |
| ring-size histogram | – | ✅ | ✅ **LLM only** | ❌ |
| top-k degree nodes (named) | – | ✅ | ✅ **LLM only** | ❌ |
| top-k betweenness nodes | – | ✅ | ✅ **LLM only** | ❌ |
| top-k eigenvector nodes | – | ✅ | ✅ **LLM only** | ❌ |
| degree_gini | ✅ | ❌ | ❌ | ❌ |
| degree_skewness | ✅ | ❌ | ❌ | ❌ |
| degree_kurtosis | ✅ | ❌ | ❌ | ❌ |
| modularity (value) | ✅ | ❌ | ❌ | ❌ |
| max_kcore | ✅ | ❌ | ❌ | ❌ |
| spectral_gap | ✅ | ❌ | ❌ | ❌ |
| n_triangles | ✅ | ❌ | ❌ | ❌ |
| n_squares | ✅ | ❌ | ❌ | ❌ |
| mean_betweenness | ✅ | ~ per-node | ❌ | ❌ |
| mean_closeness | ✅ | ❌ | ❌ | ❌ |
| mean_eigenvector | ✅ | ~ per-node | ❌ | ❌ |
| null-model contrasts (ER/config/small-world) | – | ❌ | ❌ | ❌ |

**Node-level canonical features** (closeness, pagerank, kcore, mean_neighbor_degree,
per-node clustering) — absent from facts(), verbalize(), AND fvec.

## Violations (all four must be fixed before the comparison is fair)
1. **LLM sees MORE than logreg.** verbalize() gives the LLM the ring-size histogram
   and named top-k centrality nodes; fvec gives logreg none of it. The comparison
   currently FAVORS the LLM (e.g. the scale-free prompt handed the model
   "Top brokers (betweenness)" lists logreg never sees). Not apples-to-apples.
2. **Both omit ~10 canonical features** (gini/skew/kurtosis, modularity value, kcore,
   spectral_gap, triangles, squares, closeness, mean centralities) -> both impoverished;
   "is it scale-free?" is unanswerable for either (root cause of the Holme-Kim failure
   2026-06-15: degree-distribution shape was never computed).
3. **Node-level canonical features absent everywhere.**
4. **logreg builder fragmented**: fvec for graph-cls, but crossdomain_graphcls.feat_vec,
   edge_icl.feat_vec, quick_kumo_vs_claude.feat (hardcoded subset FK), fair_node*.feats
   each roll their own. No single feature contract.

## Fix (one structural change closes all four)
Introduce a single **FEATURE_REGISTRY**: declare each feature ONCE with (compute fn,
scalar form for logreg, verbalization template). Have facts(), fvec (logreg X), and
verbalize() (LLM text) ALL derive from the registry so they cannot drift. Then:
- port the full properties.py set (gini/skew/kurtosis, modularity value, kcore,
  spectral_gap, triangles, squares, mean betweenness/closeness/eigenvector, node-level
  closeness/pagerank/kcore/neighbor-degree),
- add the null-model contrasts (ER/config baselines, small-world coefficient),
- for centrality top-k the LLM gets, also include matching SCALAR aggregates
  (mean/max/gini per centrality) in the registry so logreg gets the same information;
  named-top-k becomes presentation only,
- add a `verbalization_version` stamp emitted by facts() and recorded in every result
  manifest, so before/after (impoverished vs full) runs are never conflated.

## Null models — what we mean, and which we report
A null model is NOT unique: it is a randomized ensemble that fixes some properties and
shuffles the rest; the CHOICE of what to fix defines the question. graphlex never claims
"the" null — it reports each "beyond-chance" statistic against NAMED standard nulls,
states which, gives the ratio / z-score, and flags small-n uncertainty.

Two nulls are always constructible (need only n, m, and the degree sequence — always have):
- **ER G(n, m):** fix node + edge counts only. Asks "structure beyond density?".
  Crudest baseline.
- **Configuration model:** fix the EXACT degree sequence, randomize wiring. Asks
  "structure beyond what the degrees alone force?". The PRINCIPLED DEFAULT — it controls
  for the degree distribution, which is the main confound (hubs manufacture apparent
  clustering/assortativity).

Compute rule — **analytic where a closed form exists, else seeded degree-preserving
rewiring** (e.g. 200 double-edge swaps, fixed seed) -> mean +/- std -> z-score:
- ER clustering = density (exact).
- Config-model expected clustering (Newman): C = (<k^2> - <k>)^2 / (n <k>^3).
- Config-model expected degree-assortativity ~ 0.
- **Modularity Q is ALREADY defined vs the configuration null** — community structure
  ships with its null baked in; report Q directly.
- Small-world coefficients sigma = (C/C_rand)/(L/L_rand), omega = L_rand/L - C/C_latt.

Reporting form: "clustering 0.21; vs ER 0.13 (1.6x); vs configuration model 0.15 (1.4x);
n=30 -> wide CI." Worked example (Holme-Kim n=30, <k>=3.7, deg-std 2.3): observed 0.21,
ER 1.62x, configuration 1.37x — same direction, different magnitude, which is exactly why
the null must be NAMED. At small n the null variance is large; always emit the caveat.

## Sequencing
Hold until the running experiments finish -> implement registry once -> regenerate
prompts + logreg features -> rerun. Current results = the "before" ablation
(impoverished + LLM/logreg-mismatched verbalization).
