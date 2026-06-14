"""FAIR test (Kumo's home turf): relational NODE prediction with homophily.
Org network (SBM), predict each held-out person's department from their structural
position + neighbors' KNOWN departments. Baselines: neighbor-majority vote, and
logreg on [structural + neighbor-label-count] features. Emits the verbalize+Claude
prompt + truth."""
import sys, json, numpy as np, networkx as nx
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

rng = np.random.RandomState(1)
OUT = '/home/scratch/bench_out/verbalize_prompts'
DEPTS = ['CS', 'Math', 'Bio', 'Eng']
sizes = [30, 30, 30, 30]
# moderate homophily: informative but not trivial
P = np.full((4, 4), 0.02); np.fill_diagonal(P, 0.12)
G = nx.stochastic_block_model(sizes, P, seed=1)
lab = {}
b = 0
for k, s in enumerate(sizes):
    for _ in range(s):
        lab[b] = k; b += 1
nx.set_node_attributes(G, {v: DEPTS[lab[v]] for v in G}, 'dept')

nodes = list(G.nodes())
rng.shuffle(nodes)
n_known = int(0.70 * len(nodes))
known = set(nodes[:n_known]); held = nodes[n_known:]
clust = nx.clustering(G)

def neigh_counts(v):                     # counts over KNOWN neighbors only
    c = Counter(lab[u] for u in G.neighbors(v) if u in known)
    return [c.get(k, 0) for k in range(4)]

# ---- classical baselines ----
glob_major = Counter(lab[v] for v in known).most_common(1)[0][0]
def majority(v):
    cnt = neigh_counts(v)
    return int(np.argmax(cnt)) if sum(cnt) else glob_major
acc_major = np.mean([majority(v) == lab[v] for v in held])

def feats(v): return [G.degree(v), clust[v]] + neigh_counts(v)
Xtr = np.array([feats(v) for v in known]); ytr = np.array([lab[v] for v in known])
Xte = np.array([feats(v) for v in held]); yte = np.array([lab[v] for v in held])
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
acc_lr = float((clf.predict(sc.transform(Xte)) == yte).mean())

# ---- verbalize + Claude prompt ----
def describe(v):
    cnt = neigh_counts(v)
    cs = ", ".join(f"{DEPTS[k]}:{cnt[k]}" for k in range(4))
    return (f"- {G.degree(v)} connections, clustering {clust[v]:.2f}; "
            f"known colleagues by department: {cs}")

shot_nodes = []
by = {k: [v for v in known if lab[v] == k] for k in range(4)}
for k in range(4):
    shot_nodes += list(rng.permutation(by[k]))[:6]      # 6 per dept = 24 shots
rng.shuffle(shot_nodes)

TASK = ("Predict each person's department (one of: CS, Math, Bio, Eng) in a university "
        "collaboration network. Use the labeled examples to learn the pattern, then classify "
        "each query.\nOUTPUT FORMAT: one line per query, exactly '<id> <DEPT>'. No other text.")
lines = [TASK, "", "=== LABELED EXAMPLES ==="]
for v in shot_nodes:
    lines.append(f"[{DEPTS[lab[v]]}]\n{describe(v)}\n")
lines.append("=== QUERIES (classify each) ===")
truth = []
for q, v in enumerate(held):
    lines.append(f"Query {q}:\n{describe(v)}\n")
    truth.append([q, DEPTS[lab[v]]])
open(f"{OUT}/fairnode_verbal.txt", 'w').write("\n".join(lines))
json.dump({'fairnode': truth}, open(f"{OUT}/fairnode_truth.json", 'w'))

print(f"SBM org net: {len(nodes)} people, 4 depts; {len(known)} known, {len(held)} held-out; chance=0.250")
print(f"(1) neighbor-majority vote        : {acc_major:.3f}")
print(f"(2) logreg / struct+neighbor-feats: {acc_lr:.3f}")
print(f"(3) verbalize + Claude-ICL        : [run subagent on fairnode_verbal.txt]")
print(f"wrote fairnode_verbal.txt ({len(open(f'{OUT}/fairnode_verbal.txt').read())} chars, {len(shot_nodes)} shots)")
