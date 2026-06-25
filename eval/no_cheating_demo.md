# Classification task: ensuring the llm is not cheating

dataset: **10 synthetic graph families** described by the graphlex text + statistics. 

The model gets **3 labeled examples per class**, then classifies **50 new graphs**. (Families: random/ER, scale-free/BA, small-world/WS, random-regular, geometric, community/SBM, Holme–Kim, grid lattice, caveman, tree.)

baseline: logistic regression trained on the *same* feature vector is 100% accurate.

## What the model actually sees

A **labeled example**:
```
--- EXAMPLE 0 (Class E) ---
59 nodes, 155 edges, density 0.09 … mean degree 5.3, max 12, Gini 0.20 (mild spread) …
clustering 0.07 … 6 communities (modularity 0.36) … 97 independent cycles …
```
A **query** to classify:
```
--- QUERY 0 ---
63 nodes, 126 edges … mean degree 4.0, max 4, std 0.0, Gini 0.00 (every node identical degree) …
clustering 0.05 … diameter 5 …
```
Query 0's fingerprint — *every node has exactly degree 4* — is a random **regular** graph; the model has to work out which letter that is from the examples. (It did: `D`.)

## Result (chance = 0.10)

| method | accuracy |
|---|---|
| chance | 0.10 |
| **graphlex + LLM** — anonymized labels, in-context, **zero training** | **0.96** (48/50) |
| logistic regression on the same features (non-LLM ceiling) | 1.00 |

8 of 10 families perfect; the only 2 misses were between genuinely-overlapping families (ER↔BA and BA↔Holme–Kim — which differ only in clustering).

## The reasoning is inspectable

Before classifying, the model wrote out each *anonymized* class's structural signature — deriving the categories from the numbers, names removed:

> **H**: tree — 0 cycles, mean degree 2, very long paths, k-core 1.
> **D**: constant degree exactly 4, std 0, Gini 0 → random regular.
> **G**: zero clustering, zero triangles, positive assortativity ~0.5, long paths → grid lattice.
> **C**: sparse, hub-dominated (high skew, high max degree), low clustering → scale-free.

## The logic, in one line

Synthetic ⟹ no answer to recall. Anonymized + reshuffled labels ⟹ no name to exploit. Ten classes ⟹ not "three famous names." So **0.96 vs chance 0.10 can only come from reading the verbalized structure and learning the mapping from the examples** — genuine in-context reasoning, nearly matching the non-LLM ceiling (1.00), from prose alone.

## Honest caveats

- **Single seed.** The result is clean (only confusable-pair errors), but multi-seed before any publication claim.
- **logreg = 1.00** means these families are easily separable, so the point is **not** "LLM beats logreg" — logreg is the *ceiling*. The point is that the LLM **matches** that ceiling through **anonymized** labels, which is what proves it's reading structure rather than reciting names.

*Reproduce: `eval/anon_multiclass.py`; prompt + model answers in `bench_out/anon_multiclass/`.*
