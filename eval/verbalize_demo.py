"""Generate paste-ready few-shot prompts for the 'verbalize the structure' demo.
Two arms for the SAME query graph: (A) verbalized NetworkX features, (B) raw edge
list. Few-shot in-context: K labeled examples per class, then the query. A social
scientist pastes one of these into Claude/ChatGPT and reads off the predicted label.
"""
import sys, numpy as np, networkx as nx
sys.path.insert(0, 'src')
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

K = 5          # labeled examples per class in the prompt
SEED = 0

ds = TUDataset(root='/home/scratch/tudata', name='IMDB-BINARY')
Gs, ys = [], []
for d in ds:
    G = nx.Graph(to_networkx(d, to_undirected=True))
    if G.number_of_nodes() >= 3 and G.number_of_edges() >= 1:
        Gs.append(G); ys.append(int(d.y))
ys = np.array(ys)

def verbalize(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = [d for _, d in G.degree()]
    cl = nx.average_clustering(G)
    dens = nx.density(G)
    ncomp = nx.number_connected_components(G)
    Gc = G.subgraph(max(nx.connected_components(G), key=len))
    diam = nx.diameter(Gc) if Gc.number_of_nodes() > 1 else 0
    try: assort = nx.degree_assortativity_coefficient(G)
    except Exception: assort = float('nan')
    tri = sum(nx.triangles(G).values()) // 3
    return (f"- {n} nodes, {m} edges, density {dens:.2f}\n"
            f"- average degree {np.mean(deg):.1f} (min {min(deg)}, max {max(deg)})\n"
            f"- average clustering coefficient {cl:.2f}, {tri} triangles\n"
            f"- {ncomp} connected component(s), diameter {diam} (largest component)\n"
            f"- degree assortativity {assort:.2f}")

def edgelist(G):
    es = sorted(tuple(sorted(e)) for e in G.edges())
    return ", ".join(f"({a},{b})" for a, b in es)

rng = np.random.RandomState(SEED)
idx0 = list(rng.permutation(np.where(ys == 0)[0]))
idx1 = list(rng.permutation(np.where(ys == 1)[0]))
shots = idx0[:K] + idx1[:K]
query = idx0[K]                      # held-out query, true label = 0
true = ys[query]

TASK = ("You are classifying small social networks (ego-networks of actors who appeared "
        "together in films) into one of two genres: class A or class B. Use the patterns "
        "in the labeled examples to predict the query. Answer with just 'class A' or 'class B'.")

def build(arm):
    desc = verbalize if arm == 'verbal' else (lambda G: "edges: " + edgelist(G))
    lines = [TASK, ""]
    for j, i in enumerate(shots):
        lab = 'class A' if ys[i] == 0 else 'class B'
        lines.append(f"Example {j+1} ({lab}):\n{desc(Gs[i])}\n")
    lines.append(f"QUERY (predict the class):\n{desc(Gs[query])}\n")
    lines.append("Predicted class:")
    return "\n".join(lines)

for arm in ['verbal', 'raw']:
    print("=" * 70); print(f"ARM = {arm.upper()}   (true label = {'class A' if true==0 else 'class B'})")
    print("=" * 70); print(build(arm)); print()
