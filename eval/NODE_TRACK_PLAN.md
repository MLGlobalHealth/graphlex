# NODE_TRACK_PLAN — node-level few-shot ICL for graphlex+LLM

Lab note. Scoping + design for a NODE-level in-context-learning (ICL) track, plus
its baselines, so graphlex+LLM competes on the home turf of node/edge-level graph
foundation models (PRODIGY, OFA/One-for-All) — not just whole-graph classification.

Harness skeleton is implemented and smoke-tested: `eval/node_icl.py` (generator) +
`eval/score_node_icl.py` (scorer). This doc records the design and the open
decisions; the code is the deliverable.

## 1. Motivation — granularity-flexibility

The strategic point of this track: graphlex+LLM is **task-granularity-agnostic**.
The same `facts()` / `verbalize()` machinery used for whole-graph classification
(`eval/sweep.py`, `eval/label_curve.py`) is here pointed at a single node's k-hop
**ego-graph**. Nothing about the method changes — only *what* we verbalize (a node's
neighborhood instead of a whole graph) and *what* the prompt asks (classify the
target node instead of the whole graph). So graphlex spans **graph-level AND
node-level** in one results table, where each graph FM occupies a single row at a
single granularity:

| method                | graph-level | node-level |
|-----------------------|:-----------:|:----------:|
| graphlex + LLM (ICL)  |     yes     |    yes     |
| classical + logreg    |     yes     |    yes     |
| PRODIGY (native ICL)  |      -      |    yes     |
| OFA / One-for-All     |      -      |    yes     |
| trained GNN (GCN/GIN) |     yes     |    yes     |

PRODIGY and OFA are *node/edge*-granularity specialists; they do not do whole-graph
ICL out of the box. graphlex+LLM does both with one prompt change. That is the
table we want.

## 2. Ego-graph protocol (the key design problem)

Node-classification graphs (Cora/Citeseer/Pubmed) are far too big to dump whole into
an LLM context. So each ICL **example = the target node's local k-hop ego-graph**,
verbalized with graphlex, plus that node's label. Implemented in
`eval/node_icl.py::ego_nx` / `verbalize_ego`.

- **k (hop radius):** `KHOP = 2` by default (decision pending, see §7). 1-hop egos
  on Cora are tiny (median 3 nodes, max 10) — barely any structure for graphlex to
  describe. 2-hop egos are richer (median 12, but tail up to ~97) and contain the
  homophily signal node classifiers exploit. We cap the tail (next bullet).
- **size cap:** `MAX_EGO = 25` nodes. When a 2-hop ego exceeds the cap we keep the
  BFS-nearest nodes to the target (target always retained). This bounds prompt
  length and cost; 25 nodes verbalizes to a few hundred tokens of structure.
- **target identification:** the target is relabeled to **node id 0**, placed first,
  and the prompt states `TARGET NODE = #0` explicitly. Its own feature line is
  tagged `TARGET`. This removes any ambiguity about which node in the ego-graph is
  the one to classify.
- **structure verbalization:** `verbalize(facts(G), focus='structure')` — identical
  call to `label_curve.py`'s `family_data` (`desc = lambda G: verbalize(facts(G),
  focus='structure')`). The ego-graph is a plain undirected `networkx.Graph`.
- **node features: BOTH representations are run and compared (DECIDED).** The
  feature question (opaque word-ids vs real text) is the single biggest lever on
  whether the LLM arm beats logreg, so we generate prompt sets for BOTH and report
  the contrast (`REPS = ['opaque', 'readable']` in `node_icl.py`; per-rep prompt
  dirs `<DATASET>/{opaque,readable}/seed*_k*.txt`):

  * **arm (a) `opaque`** — Planetoid native. High-dim bag-of-words (Cora 1433,
    Citeseer 3703, Pubmed 500) with **no human-readable word labels in the
    `torch_geometric` loader**. graphlex's categorical node-attr verbalization
    (`focus='all'`, `node_attrs=['type']`, as in `proteins_data`) is built for
    *small* categorical vocabularies (atom types `t0..tk`) and would be meaningless
    here (1433 opaque categories), so BoW is **not** routed through graphlex
    node_attrs. Each node gets a compact `features [w26,w61,...]` line of its top
    `TOPW=12` active word-ids (`node_feat_summary`). Honest but opaque to the LLM.
  * **arm (b) `readable`** — REAL paper title + abstract per node, sourced and
    node-aligned in `eval/node_text.py` (see §2b below). This is the OFA-style
    "text-attributed graph" framing where the LLM reads actual paper terms; likely
    where graphlex+LLM shines. graphlex core is untouched — the readable text is
    attached the same way the opaque summary is (a per-node line appended after the
    `verbalize(facts(G))` structure block).

  Both arms share the **identical** graphlex structure verbalization; only the
  per-node feature line differs. Both arms also now show **real class NAMES** in
  the TASK header (e.g. `CLASS3 = Neural Networks`), recovered alongside the text
  (a strict improvement over the old `CLASS0..6` with no semantics — applies to
  both arms, since the names come free with the text source).

### 2b. Readable-text source (arm b) — what we wired in and how it aligns

**Source:** HuggingFace `Graph-COM/Text-Attributed-Graphs/{cora,citeseer,pubmed}/`
(`raw_texts.pt`, the Graph-LLM / CurryTang text-attributed-graph release) — per-node
`"Title: ... Abstract: ..."` strings, plus `categories.csv` for real class names.
Downloaded + cached once into `bench_out/node_icl/_textcache/<Name>.json` by
`eval/node_text.py::build_cache`. `raw_texts.pt` is a torch zip holding a single
pickled `list[str]`, read with a **restricted unpickler** (forbids all global
construction) so no arbitrary code from the external pickle runs; the BoW used for
alignment is read as a raw float32 storage out of the zip (the PyG object is never
unpickled).

**Alignment to the PyG Planetoid node ordering (the load-bearing correctness step,
verified empirically):** the TAG release is in a DIFFERENT node order than Planetoid
(and uses permuted class ids), so we cannot index-align by position.

| dataset  | alignment | how / coverage |
|----------|-----------|----------------|
| Cora     | **EXACT** | the TAG release ships the SAME binary BoW as Planetoid; recover the exact Planetoid→TAG permutation by hashing each node's binarized BoW row. **2708/2708**, 0 missing. Each node gets its OWN real title+abstract. |
| PubMed   | **EXACT** | same binarized-BoW row-match. **19717/19717**, 0 missing. (Planetoid PubMed x is TF-IDF, binarized for the match.) |
| Citeseer | **APPROX (documented fallback)** | the TAG Citeseer is a re-collection of only 3186 of Planetoid's 3327 nodes, NOT in Planetoid order, and its processed features are **SBERT-384, not the 3703-d BoW → not row-matchable**. A node-exact title is NOT cleanly obtainable for all nodes from any single release. |

**Citeseer fallback, precisely (do NOT silently keep opaque ids):** Planetoid's true
per-node class IS recoverable exactly — via a binarized-BoW row-match against the
**LINQS `citeseer.content`** original (3703-d binary BoW + class string; 3312/3327
match, 15 are zero-BoW isolates). Using that authoritative per-node class, each
Planetoid Citeseer node is assigned a **real Citeseer title+abstract drawn
deterministically (by node id) from the TAG corpus OF ITS TRUE CLASS**. So the
readable text the LLM sees is genuine in-domain Citeseer text whose **topic matches
the node's real label** (the class/homophily signal is correct), but it is **not
guaranteed to be that exact paper**. The 15 isolated nodes get a class-name stub.
This is marked `APPROX` in the manifest (`text_alignment`) and surfaced by the
scorer. Why this fallback: TAGLAS/CS-TAG ship no Citeseer; the only Citeseer text
corpus that exists is the 3186-node backup, which has no Planetoid-index bridge —
verified by two independent investigations.

**Investigated but not used:** OFA repo `cora.pt`/`pubmed.pt` (Cora+PubMed only, no
Citeseer; SBERT features); TAGLAS `WFRaain/TAG_datasets` (no Citeseer); LINQS
originals as the *text* source (only a binary word-presence vector + paper-id, no
titles/abstracts) — though LINQS IS used as the Citeseer class bridge above.

## 3. Datasets

`torch_geometric.datasets.Planetoid`, all three present and loadable at
`/home/scratch/planetoid` (Cora, Citeseer, PubMed):

| dataset  | nodes  | classes | features        | chance |
|----------|--------|:-------:|-----------------|:------:|
| Cora     | 2 708  |    7    | 1433 binary BoW | 0.143  |
| Citeseer | 3 327  |    6    | 3703 binary BoW | 0.167  |
| PubMed   | 19 717 |    3    | 500 TF-IDF      | 0.333  |

These are the canonical text/feature-attributed citation graphs — exactly the turf
PRODIGY/OFA report on. Start with Cora (smoke done); Citeseer/PubMed are one CLI arg
away (`node_icl.py Citeseer`). PubMed is large but ego-graphs + NQ queries keep cost
flat regardless of graph size.

## 4. Baseline matrix

| baseline                         | status        | how                                                    |
|----------------------------------|---------------|--------------------------------------------------------|
| trained logreg on node features  | **implemented + run** | `node_icl.py::logreg_at`, raw BoW, same K labeled nodes/class, balanced acc into manifest |
| matched few-shot trained GCN     | **implemented + run** | `node_icl.py::gcn_at` (`NodeGCN`), §4b. Same K labeled nodes/class over full graph, balanced acc on same query nodes. CPU-local. |
| graphlex+LLM, rep=opaque         | prompts ready | LLM runs later (subagents/run_qwen.py) on `<DS>/opaque/seed*_k*.txt` |
| graphlex+LLM, rep=readable       | prompts ready | same, on `<DS>/readable/seed*_k*.txt` |
| PRODIGY (native node ICL)        | **env pending — GPU** | see §6 |
| OFA / One-for-All (node ICL)     | **env pending — GPU** | see §6 |
| **GraphPFN (native node ICL)**   | **env EXISTS — likely first FM to add** | `/home/scratch/gpfn_venv` on clpc35 already has GraphPFN node mode; consumes manifest `support_ids`/`query_ids`, writes `ans/graphpfn/seed*_k*.ans`. See §6. |
| chance / majority                | in manifest   | `chance = 1/n_classes` |

The classical bar (logreg) answers "is graphlex+LLM more label-efficient than the
obvious classical thing at the same K?"; the GCN is the specialist bar (full graph +
same labels, semi-supervised). Both are **representation-independent** (they train on
the raw Planetoid feature matrix, NOT the opaque/readable text), so they are computed
ONCE per (seed,k) and stored at the manifest top level (`logreg`, `gcn`) — they apply
to both LLM arms.

### 4b. Matched few-shot GCN config (`node_icl.py::NodeGCN` / `gcn_at`)

Standard semi-supervised node GCN (2× `GCNConv`), trained transductively on the FULL
Planetoid graph but supervised by ONLY the K labeled nodes/class (identical set to
logreg/LLM), evaluated on the identical query nodes, BALANCED accuracy. Fixed config
(no per-dataset tuning): hidden 64, layers 2, dropout 0.5, Adam lr 1e-2, weight-decay
5e-4, ≤200 epochs, early-stop patience 30 on train-loss plateau (no held-out val at
K=1), class-weighted cross-entropy. CPU, <1 min/dataset (PubMed ~3 min for 3 seeds ×
3 K). It is in fact the strongest non-LLM arm here (semi-supervised use of the full
graph; see §9), not weak — reported honestly.

## 5. Seeds / K grid / scoring

- **Seeds:** `SEEDS = [11, 22, 33]` (>=3, load-bearing). Qwen can run all; Opus
  subagents anchor on the same three.
- **K (shots):** `K_SHOTS = [1, 3, 5]` labeled nodes **PER CLASS** (class-balanced,
  load-bearing). Splits are nested (K=1 support ⊂ K=3 ⊂ K=5) so the curve is clean.
  `make_splits` draws K/class shots + `NQ=30` class-balanced query nodes, disjoint
  from all shots, per seed.
- **Scoring:** **BALANCED accuracy always** (`_common.bal_acc`, load-bearing). The
  query set is class-balanced so balanced ≈ raw here, but we keep balanced for
  consistency with the rest of the suite and to stay honest if NQ doesn't divide
  evenly across classes. logreg in the manifest is stored as balanced accuracy
  (`sklearn.metrics.balanced_accuracy_score`).

## 6. How the native-ICL FM node baselines slot in (PRODIGY / OFA / GraphPFN)

**GraphPFN is the likely FIRST FM node baseline to add** — its node-mode env already
exists at `/home/scratch/gpfn_venv` on clpc35 (no new env to build, unlike PRODIGY/
OFA). It consumes the manifest `support_ids`/`query_ids` (now persisted, below),
predicts each query's class natively, and writes `<DS>/ans/graphpfn/seed*_k*.ans` in
the SAME `<id> <CLASS>` format → scored with zero scorer changes (add `'graphpfn'` to
the model loop, rep-agnostic since FMs ignore the verbalization). PRODIGY/OFA follow
once their envs are built. The design contract:

1. **Same splits — NOW PERSISTED.** All FMs consume the *exact* support/query node
   ids from `node_icl.py::make_splits`. Each manifest `files[*]` entry now carries
   `support_ids` (per-class labeled node ids, sliced to that file's K) and
   `query_ids` (the NQ query node ids), in ADDITION to `truth`. So a GPU job reads
   the identical labeled set + query set without re-deriving the split.
2. **Native ICL, no verbalization.** Unlike graphlex+LLM, PRODIGY/OFA take the raw
   graph + features + the labeled support set directly (PRODIGY: prompt graph of
   support+query; OFA: text-attributed unified node features). They do their own
   in-context prediction. We only need their per-query predicted class.
3. **Same scoring.** Have the GPU job write predictions as the SAME `<id> <CLASS>`
   answer lines into `ans/{graphpfn,prodigy,ofa}/seed*_k*.ans`. Then
   `score_node_icl.py` scores them with the identical `parse_ans` + `bal_acc` path —
   zero scorer changes; just add the model name to the loop.
4. **Checkpoints/env:** PRODIGY (MAG240M-pretrained ckpt) and OFA (its LLM-encoder +
   GNN) each need their own conda env + CUDA on clpc35. Mark as a separate GPU task;
   `/home/scratch` is not shared with clpc35 (see `run_qwen.py` header), so the
   split ids + a feature export must be shipped over, predictions shipped back.

## 7. Decisions — RESOLVED

1. **k = 1 vs 2 hop.** DECIDED: `KHOP=2` + `MAX_EGO=25` (captures homophily; tail
   capped). Re-running with k=1 is one constant change if we want the contrast later.
2. **Node features: opaque vs real text.** DECIDED: **run BOTH** (`REPS`), report the
   contrast. Real text sourced + node-aligned (§2b): EXACT for Cora/PubMed, APPROX
   (class-consistent) for Citeseer.
3. **Which datasets.** DECIDED: **all three** (Cora, Citeseer, PubMed), generated +
   baselined below.
4. **Trained-GNN baseline.** DECIDED: **added** (matched few-shot GCN, §4b).
5. **NQ / cost.** Current run `NQ=30` (→ 28 effective on Cora: `NQ//7×7`), 3 seeds ×
   3 K × 2 reps = 18 prompt files/dataset. Scale `NQ` to ~50–100 for the final table
   (cost is flat in graph size — ego-graphs + fixed NQ).

## 8. Integration points (real file/function names)

- Imports graphlex identically to `sweep.py` / `label_curve.py`:
  `sys.path.insert(0, '/home/scratch/.../graphlex'); from graphlex import facts, verbalize`.
- Prompt skeleton identical to `label_curve.py::build_prompt`: TASK header →
  `=== LABELED EXAMPLES ===` with `[CLASSx]` blocks → `=== QUERIES ===` with
  `Query <id>:` blocks → `OUTPUT FORMAT: '<id> <CLASS>'`.
- File naming: `seed{seed}_k{k}.txt`, now under a per-representation subdir:
  `/home/scratch/bench_out/node_icl/<DATASET>/<rep>/seed*_k*.txt` (`rep` ∈
  `opaque`,`readable`); answers in `.../<DATASET>/<rep>/ans/<model>/`.
- Readable text + real class names sourced by `eval/node_text.py` (§2b); cache at
  `.../node_icl/_textcache/<Name>.json`. **graphlex core untouched** — the readable
  text is appended as a per-node line after `verbalize(facts(G))`, exactly parallel
  to the opaque word-id line.
- LLM arm driven by Opus subagents / `eval/run_qwen.py` (glob `seed*_k*.txt` matches
  inside each rep dir) — **no LLM is called from `node_icl.py`**.
- Scored by `eval/score_node_icl.py` (per-rep ans dirs; reuses `_common.parse_ans` +
  `_common.bal_acc` unchanged; prints logreg + GCN + per-rep Opus/Qwen).

## 9. Results (3 seeds, balanced accuracy, mean ± std)

Generated + non-LLM baselines RUN end-to-end (`node_icl.py {Cora,Citeseer,PubMed}`),
venv `/home/scratch/fmsn-dev/.venv`. `KHOP=2`, `MAX_EGO=25`, `NQ=30`, K∈{1,3,5}/class.
logreg + GCN are representation-independent (raw Planetoid features); LLM columns
(both reps) fill in once subagents/`run_qwen.py` run on the prompt files.

**Non-LLM baselines (balanced acc):**

| dataset (chance)   | K/cls | logreg          | GCN (few-shot)   |
|--------------------|:-----:|-----------------|------------------|
| Cora (0.143)       |   1   | 0.179 ± 0.029   | **0.524 ± 0.045** |
|                    |   3   | 0.310 ± 0.045   | **0.667 ± 0.045** |
|                    |   5   | 0.393 ± 0.117   | **0.655 ± 0.073** |
| Citeseer (0.167)   |   1   | 0.267 ± 0.072   | **0.356 ± 0.057** |
|                    |   3   | 0.356 ± 0.083   | **0.467 ± 0.098** |
|                    |   5   | 0.267 ± 0.098   | **0.489 ± 0.137** |
| PubMed (0.333)     |   1   | 0.456 ± 0.057   | **0.533 ± 0.027** |
|                    |   3   | 0.456 ± 0.150   | **0.689 ± 0.042** |
|                    |   5   | 0.489 ± 0.137   | **0.722 ± 0.063** |

Notes: all arms well above chance. GCN > logreg everywhere — expected, since the GCN
is semi-supervised over the FULL graph (uses unlabeled structure), whereas logreg
sees only the K labeled feature vectors. The few-shot GCN is therefore the *strong*
non-LLM bar here, not the weak one (the "few-shot GNN is weak" intuition holds for
*inductive graph-classification* GNNs as in `gnn_baseline.py`, not for transductive
node GCNs that exploit the whole graph). Citeseer is hardest (6-class, sparse 2-hop
egos — many are 3-node as seen in the prompts), PubMed easiest (3-class).

**LLM arms (to be filled):** `Opus` / `Qwen` × `{opaque, readable}` per dataset. The
headline contrast this track exists to measure: does `readable` (real titles/
abstracts) lift the LLM above `opaque` (word-ids) and toward/over the GCN bar — the
text-attributed-graph hypothesis.

## 10. What is ready for the LLM-subagent pass

- 3 datasets × 2 reps × 3 seeds × 3 K = **54 prompt files** written, e.g.
  `bench_out/node_icl/Cora/readable/seed11_k1.txt`. Same prompt skeleton + answer
  format as every other track (`<id> <CLASS>`); real class names in the TASK header.
- Run Opus subagents (or `run_qwen.py`) over each rep dir; write answers to
  `<DATASET>/<rep>/ans/<model>/seed*_k*.ans`; score with `score_node_icl.py <DATASET>`.
- Manifests carry `support_ids` + `query_ids` so the FM node baselines (GraphPFN
  first — env already at `/home/scratch/gpfn_venv` on clpc35 — then PRODIGY/OFA) read
  the identical splits and write `ans/<fm>/…` for the same scorer.
