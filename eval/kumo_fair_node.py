"""Run REAL KumoRFM on the fair node-prediction task — same SBM graph + same
known/held split as fair_node_pred.py. KumoRFM gets the full relational graph
(nodes table with degree/clustering/dept, edges table) and a classification PQL.
Held-out depts are masked (NaN) so they can't leak into context."""
import os, json, numpy as np, networkx as nx, pandas as pd

# ---- regenerate the IDENTICAL graph + split as fair_node_pred.py ----
rng = np.random.RandomState(1)
DEPTS = ['CS', 'Math', 'Bio', 'Eng']; sizes = [30, 30, 30, 30]
P = np.full((4, 4), 0.02); np.fill_diagonal(P, 0.12)
G = nx.stochastic_block_model(sizes, P, seed=1)
lab = {}; b = 0
for k, s in enumerate(sizes):
    for _ in range(s):
        lab[b] = k; b += 1
nodes = list(G.nodes()); rng.shuffle(nodes)
n_known = int(0.70 * len(nodes)); known = set(nodes[:n_known]); held = list(nodes[n_known:])
clust = nx.clustering(G); deg = dict(G.degree()); n = G.number_of_nodes()

# sanity: held order/labels must match fairnode_truth.json (query q -> held[q])
truth = {int(q): DEPTS[lab[held[q]]] for q in range(len(held))}

# IDENTICAL features to logreg/Claude: degree, clustering, neighbor-dept counts
# (counts over KNOWN neighbors only, so held-out labels never leak).
from collections import Counter
def neigh_counts(v):
    c = Counter(lab[u] for u in G.neighbors(v) if u in known)
    return [c.get(k, 0) for k in range(4)]

cols = {"node_id": list(range(n)),
        "degree": [deg[v] for v in range(n)],
        "clustering": [clust[v] for v in range(n)]}
for k, d in enumerate(DEPTS):
    cols[f"n_{d}"] = [neigh_counts(v)[k] for v in range(n)]
cols["dept"] = [DEPTS[lab[v]] if v in known else None for v in range(n)]
nodes_df = pd.DataFrame(cols)

import kumoai.experimental.rfm as rfm
rfm.init(api_key=os.environ["KUMO_API_KEY"])
graph = rfm.LocalGraph.from_data({"nodes": nodes_df}, infer_metadata=True)
graph["nodes"]["dept"].stype = "categorical"

model = rfm.KumoRFM(graph, verbose=False)
pql = "PREDICT nodes.dept FOR EACH nodes.node_id"
res = model.predict(pql, indices=pd.Series(held), verbose=True)

print("=== raw result ===")
print(type(res))
print(getattr(res, "columns", None))
print(res.head(12) if hasattr(res, "head") else res)
res.to_csv("/home/scratch/bench_out/kumo_fair_raw.csv", index=False)

# ---- robust parse: one predicted dept per held node_id ----
pred = {}
cols = list(res.columns)
id_col = next((c for c in cols if "node_id" in str(c).lower() or str(c).upper() == "ENTITY"), cols[0])
class_cols = [c for c in cols if str(c) in DEPTS]
if class_cols:                                   # wide: a column per class (prob)
    for _, row in res.iterrows():
        pred[int(row[id_col])] = max(class_cols, key=lambda c: row[c])
else:
    cls_col = next((c for c in cols if str(c).upper() in ("CLASS", "TARGET_PRED", "PREDICTION", "DEPT")), None)
    score_col = next((c for c in cols if str(c).upper() in ("SCORE", "PROBABILITY", "PROB")), None)
    if cls_col and score_col:                    # long: rows per (node,class)
        for nid, g in res.groupby(id_col):
            pred[int(nid)] = g.loc[g[score_col].idxmax(), cls_col]
    elif cls_col:
        for _, row in res.iterrows():
            pred[int(row[id_col])] = row[cls_col]

if pred:
    held_map = {held[q]: truth[q] for q in range(len(held))}
    acc = np.mean([str(pred.get(v)) == held_map[v] for v in held if v in pred])
    print(f"\n=== KumoRFM fair node prediction: {acc:.3f} ({len(pred)}/{len(held)} predicted; chance 0.250) ===")
    print("compare: neighbor-majority 0.750 | logreg 0.667 | graphlex+Claude 0.750")
else:
    print("\n!! could not parse predictions — inspect kumo_fair_raw.csv / columns above")
