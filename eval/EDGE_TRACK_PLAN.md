# EDGE_TRACK_PLAN — edge / link-prediction few-shot ICL for graphlex+LLM

Lab note. Design + results for the THIRD task granularity: edge / link prediction.
After whole-graph classification (`eval/sweep.py`, `eval/label_curve.py`) and node
classification (`eval/node_icl.py`), this points the SAME `facts()` / `verbalize()`
machinery at a NODE PAIR `(u,v)` and asks: is there an edge between them?

Harness implemented + non-LLM baselines RUN end-to-end: `eval/edge_icl.py`
(generator + baselines) + `eval/score_edge_icl.py` (scorer). This doc records the
protocol; the code + manifests are the deliverable. Mirrors `NODE_TRACK_PLAN.md`.

## 1. Motivation — completing the granularity story

graphlex+LLM is **task-granularity-agnostic**. The same verbalization spans
graph-, node-, AND edge-level in one results table, while each specialist graph
foundation model is locked to a single granularity:

| method                       | graph-level | node-level | edge / link |
|------------------------------|:-----------:|:----------:|:-----------:|
| graphlex + LLM (ICL)         |     yes     |    yes     |   **yes**   |
| classical + logreg           |     yes     |    yes     |   **yes**   |
| trained GNN (GCN / GAE)      |     yes     |    yes     |   **yes**   |
| PRODIGY / OFA (native ICL)   |      -      |    yes     |    yes*     |
| GraphPFN (native ICL)        |      -      |    yes     |      -      |
| ULTRA (native ICL)           |      -      |     -      |   **yes**   |

ULTRA is a link-prediction-ONLY foundation model (built for multi-relational KG
completion); it does no node- or graph-level task. graphlex+LLM does all three with
one prompt change. That is the table this track completes. (*PRODIGY/OFA report
edge-level for KG-style tasks; not wired here.)

## 2. Protocol (the key design problems)

### 2a. Pair sampling + balanced query/support sets
A "query" is a node pair `(u,v)`; label = edge present (**LINK / CLASS1**) or absent
(**NOLINK / CLASS0**). Standard balanced link-prediction protocol:
- **Positive pairs** = existing undirected edges sampled from the graph.
- **Negative pairs** = true non-edges, rejection-sampled from the complement. Only
  sampled between nodes of degree ≥ 1 (so the pair actually sits in the graph and has
  a non-empty neighborhood; isolated nodes would give empty joint neighborhoods).
- **Query set** = `NQ/2` positives + `NQ/2` negatives (`NQ=40` → 20+20), BALANCED.
- **Support** = `K` positive + `K` negative example pairs, class-balanced, nested in
  K (K=1 ⊂ K=3 ⊂ K=5). `K_SHOTS=[1,3,5]`, `SEEDS=[11,22,33]` (≥3, load-bearing).
- All support + query pairs are mutually disjoint per seed (`make_splits`).

### 2b. Observed-graph split (no label leakage — load-bearing)
The positive SUPPORT edges AND positive QUERY edges are **removed** from the graph
used to (a) compute the link-feature heuristics, (b) build the verbalized
neighborhoods, and (c) train the GNN. So no method ever sees the edge it must
predict. Negatives are non-edges of the full graph (nothing to remove). This is the
standard observed-graph link-prediction setup (`observed_graph()`; the `(u,v)` edge
is also stripped defensively inside `joint_nx`). Per-seed observed graph + community
structure are computed once and reused across K and reps.

### 2c. The graphlex angle — JOINT-neighborhood verbalization
For each pair `(u,v)` we verbalize their **joint k-hop neighborhood**: the union of
`u`'s and `v`'s k-hop ego-graphs as one `networkx.Graph`, with both endpoints
relabeled to ids **#0 (u)** and **#1 (v)** and stated explicitly
(`TARGET PAIR = #0 and #1`). Structure rendered with the identical
`verbalize(facts(H), focus='structure')` call used by every other track.
- `KHOP=1` per endpoint, `MAX_JOINT=30` nodes (BFS-nearest to either endpoint kept;
  u,v always retained). 1-hop joint neighborhoods already carry the
  common-neighbor / triangle-closure signal link prediction exploits; the cap bounds
  prompt cost. (k=2 is one constant change if we want the richer-context contrast.)
- The query edge `{#0,#1}` is **never added** to `H` (it is the answer).

### 2d. Computed classical link features (appended, graphlex core untouched)
Parallel to how `node_icl.py` appends a per-node readable line, each pair's prompt
gets a **`Link features:`** line of classical link-prediction scores computed with
networkx on the OBSERVED graph (`link_feat_line`):
`common-neighbors`, `Jaccard` (`nx.jaccard_coefficient`), `Adamic-Adar`
(`nx.adamic_adar_index`), `resource-allocation` (`nx.resource_allocation_index`),
`preferential-attachment` (`nx.preferential_attachment`), `shortest-path-length`
(`nx.shortest_path_length`, sentinel "no path" if disconnected), `same-community`
(greedy-modularity). graphlex core is NOT modified — these are appended text, exactly
like the node-track readable line.

### 2e. Two representations (both run + compared)
`REPS=['opaque','readable']`, identical to the node track:
- **opaque** — Planetoid native: each node's top `TOPW=10` active word-ids `[w26,…]`.
- **readable** — real paper title+abstract per node, sourced + Planetoid-aligned by
  `eval/node_text.py` (reused verbatim): EXACT for Cora/PubMed (binarized-BoW
  row-match), APPROX/class-consistent for Citeseer (see `NODE_TRACK_PLAN.md` §2b).
Both share the identical graphlex structure verbalization + identical link-feature
line; only the per-node feature line differs.

## 3. Datasets
`torch_geometric.datasets.Planetoid` at `/home/scratch/planetoid` — Cora, Citeseer,
PubMed (same citation graphs as the node track; the turf link-prediction FMs report
on). Cost is flat in graph size: joint neighborhoods + fixed `NQ` keep prompt length
bounded even for PubMed (19.7k nodes).

## 4. Baseline matrix

| baseline                         | status        | how |
|----------------------------------|---------------|-----|
| classical-heuristic logreg       | **implemented + run** | `edge_icl.py::heuristic_logreg` — logreg on the 7-d link-feature vector (CN/Jaccard/AA/RA/PA/SP/same-comm) of the K pos + K neg SUPPORT pairs, predict the balanced query pairs. The standard link-prediction bar. Bal-acc + AUC into manifest. |
| trained GNN link predictor (GAE) | **implemented + run** | `edge_icl.py::gnn_link` — 2-layer GCN encoder + dot-product decoder, trained self-supervised on ALL observed edges (positive) + per-epoch sampled negatives, evaluated on the same query pairs. The SPECIALIST bar. **K-independent** (a dot-product decoder can't be fit from 2..10 pairs; it learns from the whole observed graph, parallel to the node-track GCN), reported identically across K rows. Bal-acc + AUC into manifest. CPU-local. |
| graphlex+LLM, rep=opaque         | prompts ready | LLM runs later via `run_opus_cli.py` / `run_qwen.py` on `<DS>/opaque/seed*_k*.txt` |
| graphlex+LLM, rep=readable       | prompts ready | same, on `<DS>/readable/seed*_k*.txt` |
| **ULTRA (native link-pred ICL)** | **env-pending / follow-on** | see §6 |
| chance                           | in manifest   | `chance = 0.5` (balanced query set) |

Both runnable baselines are representation-independent (computed on the observed
graph, not the verbalized text) → computed once per (seed,k) and stored at the
manifest top level (`heuristic_logreg`, `gnn_link`), applying to both LLM reps.

### 4b. GNN config (`edge_icl.py::GCNEncoder` / `gnn_link`)
GCN encoder `GCNConv(in→64) → relu → dropout 0.5 → GCNConv(64→32)`; decoder =
dot product of endpoint embeddings; loss = BCE over all observed edges (positive) vs
an equal count of fresh random negatives each epoch. Adam lr 1e-2, wd 5e-4, ≤200
epochs, early-stop patience 30 on train-loss plateau. Fixed config, no per-dataset
tuning. CPU; PubMed ~the slowest but minutes for 3 seeds.

## 5. Metrics
**BALANCED accuracy** (consistency with the rest of the suite; `_common.bal_acc` for
the LLM arms, `sklearn.balanced_accuracy_score` for the baselines) **AND AUC**
(`sklearn.roc_auc_score`, the standard link-prediction metric) on the balanced query
set, mean ± std over the 3 seeds. The query set is balanced so balanced-acc ≈ raw,
kept balanced for honesty + consistency. AUC is reported only for the probabilistic
baselines (logreg `predict_proba`, GNN sigmoid score); the LLM emits a hard
LINK/NOLINK label so it has no probabilistic AUC (a confidence-elicitation prompt
would be needed; out of scope) — the LLM arms are scored by balanced accuracy.

## 6. ULTRA slot-in design (env-pending / follow-on)

**ULTRA** (Galkin et al., arXiv 2310.04562; `github.com/DeepGraphLearning/ULTRA`,
public pretrained checkpoints) is THE foundation model for link prediction. It learns
*transferable relational representations* (conditioned on the graph of relation
interactions, not on fixed entity/relation embeddings) and does **zero-shot inductive
link prediction** on unseen KGs — the native-ICL analogue for the edge track.

**The adaptation problem.** ULTRA is built for **multi-relational knowledge-graph**
link prediction (triples `(head, relation, tail)`, ranking candidate tails). Our task
is **homogeneous** citation-graph link prediction (single implicit "cites" relation,
binary edge present/absent). So ULTRA needs adapting, not just running.

**Slot-in design (same query pairs, same metric):**
1. **Cast the citation graph as a 1-relation KG.** Every observed edge becomes a
   triple `(u, cites, v)` (add the inverse `(v, cited_by, u)` since ULTRA expects
   directed relations + their inverses). Relation graph is trivial (one relation +
   its inverse) but ULTRA's relation-conditioned message passing still applies — this
   is exactly the homogeneous-graph special case ULTRA's paper covers as inductive
   single-relation transfer.
2. **Same observed graph (no leakage).** Feed ULTRA the SAME per-seed observed graph
   used by every other arm (support + query positives removed). The split ids are
   persisted in the manifest (`files[*].removed_pos`, `support_pos`, `support_neg`,
   `query_pairs`) so the GPU job reconstructs the identical graph + query set without
   re-deriving the split — exactly the contract `node_icl.py` uses for GraphPFN.
3. **Scoring per query pair.** For each query `(u,v)`, score the triple
   `(u, cites, v)` with the pretrained ULTRA checkpoint (zero-shot, or
   few-shot-finetuned on the K support pairs to match the K columns). Threshold /
   rank against the negative queries to get a binary LINK/NOLINK decision.
4. **Same metric, same answer format.** Have the GPU job write predictions as the
   SAME `<id> <CLASS>` lines (`CLASS1`/`CLASS0`) into `<DS>/<rep>/ans/ultra/seed*_k*.ans`
   (rep-agnostic — the FM ignores verbalization; pick one rep dir or symlink). Then
   `score_edge_icl.py` scores them with the identical `parse_ans` + `bal_acc` path,
   and ULTRA's raw triple scores give a real AUC. **Zero scorer changes**; just add
   `'ultra'` to the model loop.
5. **Env/checkpoint.** ULTRA needs its own conda env + CUDA (`torchdrug`/PyG stack)
   on a GPU box (e.g. clpc35); public `ultra_3g`/`ultra_4g`/`ultra_50g` checkpoints.
   `/home/scratch` is not shared with clpc35, so ship the persisted split ids + a
   feature/edge export over, run ULTRA, ship the `.ans` files back. Marked
   **env-pending / follow-on** — NOT installed in this run, only specced here.

## 7. Seeds / K grid / cost
- **Seeds:** `[11,22,33]` (≥3, load-bearing).
- **K:** `[1,3,5]` pos + neg pairs per class (class-balanced, nested).
- **NQ:** 40 query pairs (20 pos + 20 neg). Per dataset: 2 reps × 3 seeds × 3 K = 18
  prompt files → **54 across the three datasets**. Scale `NQ` to ~100 for the final
  table (cost flat in graph size).

## 8. Integration points (real file/function names)
- Imports graphlex identically to every track:
  `sys.path.insert(0, '/home/scratch/.../graphlex'); from graphlex import facts, verbalize`.
- Prompt skeleton identical to the other tracks: TASK header →
  `=== LABELED EXAMPLES ===` with `[CLASS1]`/`[CLASS0]` blocks → `=== QUERIES ===`
  with `Query <id>:` blocks → `OUTPUT FORMAT: '<id> <CLASS>'`. Query order is
  shuffled (fixed seed) so label can't be read off position.
- File naming: `seed{seed}_k{k}.txt` under a per-rep subdir
  `/home/scratch/bench_out/edge_icl/<DS>/<rep>/seed*_k*.txt`; answers in
  `.../<DS>/<rep>/ans/<model>/`. **The existing drivers work UNCHANGED**:
  `run_opus_cli.py` / `run_qwen.py` glob `ROOT/*/seed*_k*.txt` and write
  `ROOT/<rep>/ans/<model>/<stem>.ans`, and `_common.parse_ans` reads the
  `<id> CLASS1/CLASS0` lines as-is.
- Readable text reused from `eval/node_text.py::load_readable` (no change).
- `_common.parse_ans` + `_common.bal_acc` reused unchanged by `score_edge_icl.py`.
- **No LLM is called from `edge_icl.py`.**

## 9. Results (3 seeds, balanced accuracy + AUC, mean ± std)

Generated + both non-LLM baselines RUN end-to-end
(`edge_icl.py {Cora,Citeseer,PubMed}`), venv `/home/scratch/fmsn-dev/.venv`.
`KHOP=1`, `MAX_JOINT=30`, `NQ=40`, K∈{1,3,5} pos+neg pairs/class, chance 0.500.
Baselines are representation-independent (computed on the observed graph). GNN is
K-independent (trained on all observed edges). LLM columns fill in once the drivers
run on the prompt files.

**heuristic-logreg (balanced acc / AUC):**

| dataset  | K | bal-acc        | AUC            |
|----------|:-:|----------------|----------------|
| Cora     | 1 | 0.783 ± 0.047  | 0.886 ± 0.043  |
|          | 3 | 0.792 ± 0.042  | 0.879 ± 0.033  |
|          | 5 | 0.792 ± 0.031  | 0.878 ± 0.028  |
| Citeseer | 1 | 0.608 ± 0.042  | 0.558 ± 0.159  |
|          | 3 | 0.700 ± 0.020  | 0.621 ± 0.103  |
|          | 5 | 0.633 ± 0.077  | 0.642 ± 0.019  |
| PubMed   | 1 | 0.825 ± 0.108  | 0.967 ± 0.005  |
|          | 3 | 0.825 ± 0.108  | 0.927 ± 0.044  |
|          | 5 | 0.842 ± 0.118  | 0.916 ± 0.082  |

**trained GNN link predictor / GAE (balanced acc / AUC), K-independent:**

| dataset  | bal-acc        | AUC            |
|----------|----------------|----------------|
| Cora     | 0.792 ± 0.059  | 0.887 ± 0.013  |
| Citeseer | 0.708 ± 0.012  | 0.870 ± 0.053  |
| PubMed   | 0.750 ± 0.000  | 0.932 ± 0.022  |

Notes: all arms well above the 0.500 chance bar. heuristic-logreg is already strong
on Cora/PubMed (the citation-graph link signal is largely structural — common
neighbors + Adamic-Adar — which the 7-d feature vector captures from just K support
pairs). The GAE matches/edges it on bal-acc and is the stronger, more stable AUC bar
on Citeseer (the heuristics' AUC there is noisy: sparse 1-hop joint neighborhoods
mean many pairs share zero common neighbors, flattening the heuristic scores). PubMed
heuristics have very high AUC but high bal-acc variance (the fixed 0.5 logreg
threshold is mis-calibrated on some seeds despite excellent ranking). Citeseer is
hardest (sparsest graph → least neighborhood signal), as in the node track.

**LLM arms (to be filled):** `Opus` / `Qwen` × `{opaque, readable}` per dataset. The
headline contrast: does the joint-neighborhood verbalization + computed link-feature
line let the LLM read off the link signal, and does `readable` (real titles/abstracts
→ topical/semantic similarity of the pair) lift it above `opaque` toward/over the GNN
and heuristic bars?

## 10. What is ready for the LLM-subagent pass
- 3 datasets × 2 reps × 3 seeds × 3 K = **54 prompt files** written, e.g.
  `bench_out/edge_icl/Cora/readable/seed11_k1.txt`. Same prompt skeleton + answer
  format (`<id> CLASS1/CLASS0`) as every other track.
- Run `run_opus_cli.py /home/scratch/bench_out/edge_icl/<DS>` (or `run_qwen.py`) over
  each dataset; answers land in `<DS>/<rep>/ans/<model>/seed*_k*.ans`; score with
  `score_edge_icl.py <DS>`. Drivers UNCHANGED.
- Manifests persist `support_pos` / `support_neg` / `query_pairs` / `removed_pos` so
  ULTRA (the link-pred FM, §6) reads the identical observed graph + query set and
  writes `ans/ultra/…` for the same scorer.
