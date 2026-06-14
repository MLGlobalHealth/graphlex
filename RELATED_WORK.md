# Related work — what graphlex must be positioned against

Graph -> natural language for LLMs is an **active, well-cited field**. graphlex is
a **tool + position paper**, NOT the first to verbalize graphs for LLMs. Read the
closest works in full before fixing the framing.

## Closest prior work (read in full)
- **Talk like a Graph** — Fatemi, Halcrow, Perozzi (Google, ICLR 2024).
  arXiv:2310.04560. First comprehensive study of *encoding graphs as text* for
  LLMs; introduces the GraphQA benchmark; encoding choice swings performance
  4.8–61.8%. Overlaps "how you present a graph to an LLM matters."
- **GraphText** — Zhao et al., 2023. arXiv:2310.01089. Graph -> NL via a
  graph-syntax tree of node attributes + relations; **matches/surpasses trained
  GNNs via in-context learning, no training.** *Closest to our thesis — pre-empts
  "verbalize -> ICL ≈ GNN." Read this most carefully.*
- **NLGraph** — Wang et al. ("Can LLMs solve graph problems in natural
  language?"). 29k graph-reasoning problems in text; LLMs degrade on complex
  tasks and **spurious correlations** — consistent with our permutation result
  (raw-edge ICL collapses to chance when node-ordering artifacts are removed).
- Surveys: "Graph Learning in the Era of LLMs" (arXiv:2412.12456), "A Survey of
  Graph Meets Large Language Model" (IJCAI 2024).

## graphlex's defensible niche (vs the above)
1. **Deterministic COMPUTED-FEATURE verbalization** (centralities, communities,
   null-model comparisons, homophily / structure×attributes) with **no LLM in the
   rendering** — prior work mostly encodes *raw* structure (edge lists/adjacency)
   and asks the LLM to compute. We argue the inverse and show why it's necessary.
2. **The controlled permutation experiment**: raw-structure ICL -> chance,
   verbalized-features ICL -> 0.92. A crisp argument *for* feature-verbalization
   over structure-encoding.
3. **Social-science framing**: structure × attributes + null models +
   interpretability for small networks (an audience the ML-benchmark work skips).
4. **A practical, deterministic, NetworkX-native toolkit + agent skills (MCP)** —
   a software contribution, not only a study.

## Honest caveat
GraphText already showed verbalize+ICL can match GNNs, so that part of the thesis
is NOT novel. The novelty must rest on (1)–(4) above. Do the full related-work
pass before committing the paper's contribution claims.
