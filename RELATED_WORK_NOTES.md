# Related-work notes (full reads, 2026-06-14)

Source: each paper read in full (abstract + methods + results) via WebFetch on
arXiv abstract pages, PDFs, and ar5iv HTML. Per paper: (a) what they do, (b) what
they *encode* (raw structure vs computed features vs attributes), (c) key result,
(d) precisely where graphlex differs. The honest verdict on novelty is at the end.

> Bottom line up front: GraphText pre-empts the **abstract** thesis ("deterministic
> verbalization, no LLM in the rendering, lifts ICL toward GNN parity"). It does
> **not** pre-empt graphlex's specific move: verbalizing **computed graph-theoretic
> statistics + null-model comparisons** in **prose**, for **small attributed
> social networks**. Novelty must be claimed there, not on "verbalize → ICL works."

---

## 1. GraphText — Zhao et al. 2023, arXiv:2310.01089  *(closest competitor; read most carefully)*

(a) **What they do.** For each target node, sample a k-hop ego-subgraph, build an
ordered DAG ("graph-syntax tree" / G-Syntax-Tree) whose internal nodes are
attribute-type / relation-type labels (center-node, 1st-hop, 2nd-hop, label,
feature) and whose leaves are per-node text attributes, then topologically
traverse the tree to emit a 1-D token sequence. Rendering is deterministic
(template traversal, no LLM). Reasoning is either training-free ICL with
ChatGPT/GPT-4 **or** LoRA-tuned Llama-2-7B (continuous features via an MLP
projector / new vocab tokens).

(b) **What it encodes.** Two ingredients: (i) node text attributes — raw labels,
K-means-discretized features, and **propagated** features/labels AᵏX, AᵏY (the GNN
message-passing prior); (ii) relations as |V|×|V| matrices: adjacency,
shortest-path-distance, feature-similarity XXᵀ, Personalized PageRank (α=0.25).
**It does NOT compute graph-theoretic summary statistics** — verified zero
mentions of modularity, assortativity, clustering coefficient, community
detection, or null/configuration-model comparison. PPR is used only as a
neighbor-ordering relation, not as a verbalized centrality. Output is bracketed
XML-ish lists (`<center node>['A']</center node> <ppr>['A','B','A']</ppr>`), **not
narrative prose**. "Social" never appears; datasets are citation/webpage graphs.

(c) **Key result.** Datasets: Cora, Citeseer + heterophilic WebKB (Texas,
Wisconsin, Cornell). Training-free ICL best numbers: Cora 68.3, Citeseer 58.6,
Texas 75.7, Wisconsin 54.9, Cornell 51.4. vs GCN 81.4/69.8/59.5/49.0/37.8.
→ **ICL beats GNNs on the three heterophilic WebKB sets and at low label rates,
but trails badly on Cora/Citeseer.** Matching GNNs on the text-attributed sets
requires LoRA fine-tuning (Llama-2-7B-feat 87.1/74.8). Also claims interpretable +
interactive reasoning (Sec 4.2) — but anecdotal (one Cora node, 73.3→100 after
human feedback).

(d) **Where graphlex differs.** (i) graphlex verbalizes **computed global/positional
statistics** (centralities, communities/modularity, **assortativity/homophily**,
**null-model deltas**) — GraphText verbalizes only raw + propagated node
attributes and neighbor labels, never derived global statistics or null baselines.
(ii) graphlex is **prose**, training-free end-to-end (no LoRA, no MLP projector,
no feature engineering like K-means discretization). (iii) graphlex targets
**small attributed social networks + structure×attributes**, absent here.
**Honest pre-emption:** the generic claim "deterministic template verbalization
(no LLM in rendering) makes ICL GNN-competitive" is GraphText's; and
"interpretability via NL graph reasoning" is also already claimed. graphlex must
differentiate on *what* is verbalized and on the *social-science / null-model*
framing — NOT on the deterministic-verbalize-enables-ICL idea itself.

---

## 2. Talk like a Graph — Fatemi, Halcrow, Perozzi (Google), ICLR 2024, arXiv:2310.04560

(a) **What they do.** Introduce the **GraphQA** benchmark; study how to encode a
graph as text for frozen LLMs (PaLM 2 family). Factor the prompt into a
graph-encoder g(G) × question-rephraser q(Q). Compare 9 encoders (Adjacency,
Incident, Co-authorship, Friendship, Social-network, Politician, GOT, South-Park,
Expert) and node-ID schemes (integer / English names / letters) across 7 tasks
(edge existence, node degree, node/edge count, connected nodes, cycle check,
disconnected nodes).

(b) **What it encodes.** **Raw structure only** — every encoder is a different
natural-language *skin* over the same edge set (adjacency or incident lists); the
**LLM is asked to compute the answer itself**. No centrality, community,
clustering, assortativity, or null-model is ever computed-and-verbalized. No node
attributes (pure topology). Graphs are synthetic (ER, BA, SFN, SBM; star/path/
complete), all 5–20 nodes.

(c) **Key result.** Encoder choice swings accuracy **4.8%–61.8%** (range across
tasks); e.g. connected-nodes zero-shot Adjacency 19.8% vs Incident 53.8% (~34 pt
swing). LLMs frequently **below the majority-class baseline** (edge existence
never beats majority). "Simple prompts for simple tasks" — zero-shot-CoT *hurts*
basic tasks; CoT/few-shot help only on harder/multi-hop tasks. Sparser graphs
(star, path) are easier; **disconnected-nodes ~0.5% → "LLMs lack a global model
of a graph."**

(d) **Where graphlex differs.** Their entire design space is "skin the raw edge
list, let the LLM run the algorithm"; graphlex's move — *deterministically compute*
the structure and verbalize *that* — is exactly the step they never take. Their
own negative results (sub-majority accuracy; "lack a global model";
disconnected-nodes ≈0%) are **strong motivation** for graphlex. graphlex also uses
real small social networks + attributes + null models, all absent here. **Pre-emption
to acknowledge:** they already established that (i) verbalization/encoding choice
massively swings LLM graph performance, and (ii) social-flavored vs integer node
labels matter — so graphlex cannot claim "phrasing matters" or "social labels" as
novel; only "verbalize *computed* statistics." Their named-vs-integer ablation is
the nearest thing to a labeling control but is **not** a permutation robustness test.

---

## 3. NLGraph — Wang et al., NeurIPS 2023, arXiv:2305.10037 (confirmed ID)

(a) **What they do.** First systematic NL graph-reasoning benchmark: 5,902
standard / 29,370 extended problems over 8 algorithmic tasks (connectivity, cycle,
topological sort, shortest path, max flow, bipartite matching, Hamilton path, GNN
message-passing simulation), difficulty tiered by node count. Propose Build-a-Graph
and Algorithmic prompting. LLMs: text-davinci-003, gpt-3.5/4, code-davinci-002.

(b) **What it encodes.** **Raw edge lists in prose** ("(i,j) means node i and node
j are connected…"), integer nodes, weights for weighted tasks. **No computed/
derived features verbalized** — exactly the regime graphlex argues fails. No
attributes, no social networks.

(c) **Key result.** On simple tasks CoT/self-consistency beats random by
37.3–57.8%; on complex tasks CoT becomes counterproductive (few-shot can
underperform zero-shot). **Spurious-correlation finding (load-bearing for
graphlex):** adversarial "chain" (degree-1 endpoints that ARE connected) and
"clique" (high-degree, frequently-mentioned nodes that are NOT connected) cases
drop accuracy >40% / 10–13.8%. Their words: *"LLMs might just be counting node
mentions instead of actually finding paths … LLMs are indeed (un)surprisingly
vulnerable to spurious correlations in structured reasoning."*

(d) **Where graphlex differs.** graphlex is about verbalizing *computed features*
of *attributed social networks* + graph-family ID, not algorithmic puzzles.
**Use of this paper:** cite as **convergent, not identical** support. NLGraph
shows brittleness to a *related* surface cue (degree/mention frequency); it never
runs graphlex's specific control (permute node labels to strip ordering artifacts).
Frame as "consistent with" — graphlex isolates node-ordering/label artifacts and
shows raw-edge accuracy collapses to chance under permutation while verbalized
features hold ~0.92.

---

## 4. Survey — "Graph Learning in the Era of LLMs" (Li, Wu, et al.), arXiv:2412.12456 (Dec 2024, preprint)

(a) **What it is.** Survey from a Data / Models / Tasks perspective, centered on
Text-Attributed Graphs (TAGs). Models axis = five collaboration modes: independent
collaborators; GNN-enhanced LLM; LLM-enhanced GNN; GNN-only; LLM-only — emphasis
on *learnable / joint-training* integration.

(b)–(c) **Taxonomy + where graphlex sits.** graphlex = "LLM-only" + graph-to-text
encoding for training-free ICL — acknowledged but a *sub-mode*, not a first-class
category; the field is framed as embedding-injection / fine-tuning dominated.
Its graph-to-text exemplars verbalize **raw topology** (edge lists / syntax trees),
**not computed features** — so "verbalize computed network statistics" is **not a
recognized category** (a genuine gap to claim). Open problems it names that map to
graphlex: interpretability via NL explanation, limited generalization of
training-free reasoning, hallucination (cites InstructGraph's alignment fix),
few-/zero-shot on limited graph data, efficiency. Closest cited cluster:
GraphText, Talk-like-a-Graph, NLGraph, GPT4Graph, InstructGLM — all verbalize
structure/topology, none verbalize computed social-science statistics.
*(Flag: gap phrasings read via summarizer; verify wording before direct quote.)*

(d) **Where graphlex differs.** Provides the "no recognized category for computed-
feature verbalization" gap statement, plus the training-free + interpretability +
hallucination open problems graphlex is designed around.

---

## 5. Survey — "A Survey of Graph Meets Large Language Model" (Li, Li, et al.), IJCAI 2024, arXiv:2311.12399

(a) **What it is.** IJCAI-2024 survey-track; organizes the field by the **role the
LLM plays**: LLM-as-Enhancer (explanation-based: TAPE/KEA/LLMRec; embedding-based:
GIANT/SimTeG/WalkLM/OFA), LLM-as-Predictor (flatten-based graph→text: NLGraph,
GraphText, GPT4Graph, InstructGLM, GIMLET; GNN-based: GraphGPT/GraphLLM/MolCA),
GNN-LLM Alignment (contrastive/iterative/distillation), plus Others.

(b)–(c) **Taxonomy + where graphlex sits.** Closest to **flatten-based
LLM-as-Predictor / graph-as-text**, but the survey frames flattening as feeding
*raw topology* (adjacency/edge arrows/node text) to the LLM to predict.
**No dedicated treatment of computed graph properties as verbalized features** —
graphlex's angle is a genuine gap (flag: absence-of-category inference, not an
explicit claim). Verbatim future directions that help graphlex: **data leakage** —
*"LLMs may have seen and memorized at least part of the test data of common
benchmark datasets, especially for citation networks"* → need "fair, systematic,
comprehensive benchmark"; **non-text/structural graphs** — *"a great deal of it
lacks rich textual information"*; **explainability** — LLMs "exhibit improved
explainability compared to GNNs … by offering reasoning processes"; **efficiency**
(no-GPU stance maps here); **expressive power** — "How effectively do LLMs
understand graph structures?" (vs Weisfeiler-Lehman). Closest "verbalize" methods:
GPT4Graph (GML/GraphML), GraphText (syntax trees), NLGraph (adjacency lists),
Talk-like-a-Graph (11 encodings), InstructGLM, WalkLM (random-walk textualization),
OFA — all raw structure/attributes; none verbalize computed statistics with no LLM
in rendering.

(d) **Where graphlex differs.** Supplies the **data-leakage / memorization**
caveat (critical — see risks below) and the explainability/efficiency/expressive-
power framing. Note the survey does **not** name "hallucinated structure" or
"small social networks" as directions — graphlex should own those, not cite the
survey for them.

---

## Honest synthesis: what survives as novel

**Does NOT survive (pre-empted — do not claim):**
- "Deterministic verbalization with no LLM in the rendering enables training-free
  ICL competitive with GNNs." → **GraphText.**
- "How you present a graph to an LLM matters a lot." → **Talk like a Graph.**
- "LLMs exploit surface/shortcut artifacts instead of reading structure." →
  **NLGraph** (degree/mention shortcut) — graphlex's permutation result is a
  cleaner *instance*, not the first observation.
- "Interpretability via natural-language graph reasoning." → GraphText (anecdotal)
  + both surveys list it as a known direction.

**Survives as defensible novelty (claim these, narrowly):**
1. **Verbalizing COMPUTED graph-theoretic statistics** (centralities, communities/
   modularity, assortativity/homophily, **and explicit null-model comparisons**)
   in **prose** — confirmed *not done* by GraphText, Talk-like-a-Graph, NLGraph, or
   named in either survey's taxonomy. This is the real wedge.
2. **The clean permutation control** as a crisp, isolated demonstration that
   raw-edge ICL reads node-ordering artifacts (→ chance under relabeling) while
   computed-feature verbalization is invariant (~0.92). NLGraph is convergent
   support, not the same experiment; GraphText/Talk-like-a-Graph never run it.
3. **Structure×attributes + null-model framing for small social networks** — an
   audience and an analysis style (homophily, group mixing, who occupies central
   positions, structure-as-finding-vs-null) that the ML-benchmark literature skips.
4. **A deterministic, interpretable, NetworkX-native toolkit + agent skills (MCP)**
   — a software/position contribution; the inspectable versioned thresholds table
   (numbers→words, no generative model) is a concrete interpretability artifact
   that GraphText's anecdotal NL-trace interpretability does not provide.

**The biggest risk the reads surfaced (NEW):** *benchmark memorization.* IJCAI
survey states verbatim that LLMs likely memorized common benchmarks (esp. citation
networks). This threatens graphlex's headline synthetic result, whose own takeaway
admits "Claude has priors about ER/BA/WS" — i.e. part of the 0.92 may be *prior/
memorization*, not feature-reading. It also rules out famous networks (Zachary
karate club, Cora) as clean real-data tests. Must be addressed head-on in the
experiments (see PAPER_PLAN.md): novel/renamed real networks, leakage probes, and
separating "reads the verbalized features" from "recognizes a known distribution."
