# graphlex — experimental results (2026-06-14)

All experiments validate the thesis: **verbalize computed structure -> LLM
in-context learning** beats raw-structure prompting and is competitive with
classical methods and KumoRFM on small networks.

> Method note: the "LLM" arm was run with **Claude Code subagents** as a stand-in
> for a social scientist's chatbot — each subagent is a fresh Claude given the
> pasted prompt, **no tools, pure in-context reasoning** (verified: 0 tool uses).
> Scripts here currently depend on the `fmsn-dev` data + venv (TUDataset loaders,
> KumoRFM embeddings, `graphlex` on PYTHONPATH). **All numbers are SINGLE-RUN,
> small, and mostly synthetic — replicate multi-seed before publishing.**

## 1. Synthetic graph-family identification (ER vs BA vs WS)
`make_verbalize_eval.py` — 24 queries, chance 0.333, 18-shot context.

| arm | acc |
|---|---|
| logreg on features (reference) | 0.708 |
| **verbalized features + Claude** | **0.917** |
| raw edge list + Claude | 0.875 *(artifact: generators leak family via node indexing)* |
| **raw edge list + Claude, node labels permuted** | **0.333 (chance)** |

Takeaway: the LLM cannot read structure from a raw edge list once node-ordering
artifacts are removed; verbalize the computed features and it beats logreg
(Claude has priors about ER/BA/WS).

## 2. Quick: IMDB-BINARY graph classification
`quick_kumo_vs_claude.py` — 20 shots, 40 queries, chance 0.500.

| predictor | acc |
|---|---|
| logreg / KumoRFM embeddings | 0.475 |
| logreg / graphlex features | 0.550 |
| **graphlex-verbalize + Claude-ICL** | **0.650** |

## 3. Fair: relational node (department) prediction — Kumo's home turf
`fair_node_pred.py` (baselines + prompt), `kumo_fair_node.py` (live KumoRFM).
SBM org network, 36 held-out, chance 0.250.

| method | acc |
|---|---|
| logreg / struct+neighbor features | 0.667 |
| neighbor-majority vote | 0.750 |
| **graphlex-verbalize + Claude-ICL** | **0.750** |
| KumoRFM (identical tabular features) | 0.639 |
| KumoRFM (native relational: edges+links) | 0.306 ≈ chance — see caveat |

**Kumo native-mode caveat:** the FK links register correctly (not a bug). The
controlled `kumo_linktest.py` (predict dept from neighbors' *non-target* feature,
own feature hidden) scored 0.139 — so neighbor-feature aggregation is not
happening across the self-edges table. Most likely the task was expressed wrong
for Kumo's multi-table schema. **Do not report native-mode as Kumo's real
performance**; the fair Kumo number is the tabular 0.639. Open question for the
Kumo team: correct schema for homogeneous-graph node/collective classification.

## Reproduce
Needs the fmsn-dev environment. KumoRFM scripts need `KUMO_API_KEY`
(`/home/scratch/.kumo_api_key`, free tier = 1000 req/day). Subagent arms were run
interactively via Claude Code; prompts are written to
`/home/scratch/bench_out/verbalize_prompts/`.
