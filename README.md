# graphlex

Turn a [NetworkX](https://networkx.org) graph into **exact, deterministic,
LLM-interpretable language** — so a person (or an LLM agent) can reason about a
network without an embedding, a GPU, or any risk of the model hallucinating
structure it cannot actually compute.

Built *on top of* NetworkX (BSD-3-Clause); not a fork. Operates on plain
`nx.Graph` objects.

## Why

An LLM cannot read graph structure off a raw edge list — randomly relabel the
nodes and its accuracy on "is this network random / scale-free / small-world?"
collapses to chance. But hand it the structure **computed and written in words**
and it not only works, it beats a trained classifier. So: NetworkX computes,
graphlex verbalizes (deterministically, no LLM), the agent interprets.

## Design rule (the important one)

`facts()` and `verbalize()` contain **NO LLM**. `facts()` computes exact
quantities with NetworkX; `verbalize()` renders them through templates +
inspectable threshold tables. The only "interpretation" is a versioned config
(e.g. assortativity `>= 0.6` -> "strongly homophilous"), never a generative
model. An LLM, if present, sits in the *agent* layer and reasons over these
deterministic facts — it is never the source of a quantitative claim.

## Two first-class modes (both are legitimate science)

1. **Structure-as-finding** — is it small-world / scale-free / modular /
   core-periphery? (reported relative to a null model, which is what makes it a
   finding).
2. **Structure x attributes** — homophily, group mixing, and which kinds of
   nodes occupy which structural positions.

## Quickstart

```python
import networkx as nx
from graphlex import facts, verbalize

G = nx.karate_club_graph()
print(verbalize(facts(G, node_attrs=["club"]), focus="all"))
```

See `examples/office.py` for the structure x attributes worked example.

## Status

v0.0.1 — minimal core (`facts`, `verbalize`) + the office example. Roadmap:
null-model baselines, community-stability, agent skills + MCP adapter, eval
harness (verbalize-vs-raw, permutation control, vs-logreg).
