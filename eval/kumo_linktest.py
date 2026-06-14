"""Controlled test: are the FK links FUNCTIONALLY used for NON-target neighbor
features? Predict each held-out node's dept, but its OWN dept-correlated feature
(`cohort`) is hidden (NaN); only KNOWN neighbors carry cohort. Success is only
possible by aggregating neighbors' cohort through the links.
  >> chance  -> links work for non-target features; native 0.306 was target-exclusion
  ~ chance   -> links/aggregation not working (deeper issue)
"""
import os, numpy as np, networkx as nx, pandas as pd
import kumoai.experimental.rfm as rfm

rng = np.random.RandomState(1)
DEPTS = ['CS', 'Math', 'Bio', 'Eng']; COLOR = ['red', 'green', 'blue', 'yellow']
sizes = [30, 30, 30, 30]; P = np.full((4, 4), 0.02); np.fill_diagonal(P, 0.12)
G = nx.stochastic_block_model(sizes, P, seed=1)
lab = {}; b = 0
for k, s in enumerate(sizes):
    for _ in range(s):
        lab[b] = k; b += 1
nodes = list(G.nodes()); rng.shuffle(nodes)
known = set(nodes[:84]); held = list(nodes[84:]); n = G.number_of_nodes()
truth = {int(q): DEPTS[lab[held[q]]] for q in range(len(held))}

ndf = pd.DataFrame({
    "node_id": list(range(n)),
    "degree": [G.degree(v) for v in range(n)],
    "clustering": [nx.clustering(G, v) for v in range(n)],
    # cohort = dept-color, present ONLY for known nodes -> held-out must use neighbors'
    "cohort": [COLOR[lab[v]] if v in known else None for v in range(n)],
    "dept": [DEPTS[lab[v]] if v in known else None for v in range(n)],
})
edges = list(G.edges())
edf = pd.DataFrame({"edge_id": list(range(len(edges))),
                    "source_id": [u for u, _ in edges], "target_id": [v for _, v in edges]})

rfm.init(api_key=os.environ["KUMO_API_KEY"])
g = rfm.LocalGraph.from_data({"nodes": ndf, "edges": edf}, infer_metadata=True)
g.link(src_table="edges", fkey="source_id", dst_table="nodes")
g.link(src_table="edges", fkey="target_id", dst_table="nodes")
g["nodes"]["dept"].stype = "categorical"; g["nodes"]["cohort"].stype = "categorical"
print("registered links:", g.edges)

model = rfm.KumoRFM(g, verbose=False)
res = model.predict("PREDICT nodes.dept FOR EACH nodes.node_id", indices=pd.Series(held), verbose=True)
pred = {int(nid): grp.loc[grp["SCORE"].idxmax(), "CLASS"] for nid, grp in res.groupby("ENTITY")}
acc = np.mean([pred[held[q]] == truth[q] for q in range(len(held)) if held[q] in pred])
print(f"\n=== LINK TEST: predict dept from NEIGHBORS' cohort via links (own cohort hidden): {acc:.3f} (chance 0.250) ===")
print(">> chance => links functionally used for non-target features; native 0.306 was target-exclusion.")
