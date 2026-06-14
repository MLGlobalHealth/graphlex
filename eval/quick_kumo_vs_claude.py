"""Quick head-to-head on IMDB-BINARY graph classification, SAME split / SAME shot
budget for all three:
  (1) logreg on graphlex structural features   (classical)
  (2) logreg on KumoRFM embeddings             (Kumo's representation)
  (3) graphlex-verbalize + Claude-ICL          (built separately via subagent)
Emits the few-shot prompt for (3) + ground truth, and prints (1)/(2)."""
import sys, json, numpy as np, networkx as nx
sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
sys.path.insert(0, 'src')
from graphlex import facts, verbalize
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

K, M = 10, 20                       # per class: 10 shots, 20 queries (-> 20 shots, 40 queries)
rng = np.random.RandomState(0)
OUT = '/home/scratch/bench_out/verbalize_prompts'

ds = TUDataset(root='/home/scratch/tudata', name='IMDB-BINARY')
Gs, ys = [], []
for d in ds:
    G = nx.Graph(to_networkx(d, to_undirected=True))
    if G.number_of_nodes() >= 3 and G.number_of_edges() >= 1:
        Gs.append(G); ys.append(int(d.y))
ys = np.array(ys)

i0 = list(rng.permutation(np.where(ys == 0)[0])); i1 = list(rng.permutation(np.where(ys == 1)[0]))
shot_idx = i0[:K] + i1[:K]
query_idx = i0[K:K + M] + i1[K:K + M]; rng.shuffle(query_idx)

FK = ['n_nodes', 'n_edges', 'density', 'mean_degree', 'max_degree', 'avg_clustering', 'n_communities']
def feat(G): s = facts(G)['structure']; return [s[k] for k in FK]

def logreg(Xs, ysh, Xq, yq):
    sc = StandardScaler().fit(Xs)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xs), ysh)
    return float((clf.predict(sc.transform(Xq)) == yq).mean())

ys_s, ys_q = ys[shot_idx], ys[query_idx]
# (1) graphlex features
Xs = np.array([feat(Gs[i]) for i in shot_idx]); Xq = np.array([feat(Gs[i]) for i in query_idx])
acc_feat = logreg(Xs, ys_s, Xq, ys_q)
# (2) kumorfm embeddings (same loader order -> same indexing)
E = np.load('/home/scratch/real_fm_embeddings/kumorfm__imdb.npz')['pooled']
acc_kumo = logreg(E[shot_idx], ys_s, E[query_idx], ys_q) if E.shape[0] == len(ys) else float('nan')

# (3) build verbalize+Claude-ICL prompt
LET = {0: 'A', 1: 'B'}
TASK = ("Classify each movie-collaboration ego-network as genre A or genre B. Learn the "
        "pattern from the labeled examples, then classify each query.\n"
        "OUTPUT FORMAT: one line per query, exactly '<id> <LETTER>'. No other text.")
lines = [TASK, "", "=== LABELED EXAMPLES ==="]
for i in shot_idx:
    lines.append(f"[genre {LET[ys[i]]}]\n{verbalize(facts(Gs[i]), focus='structure')}\n")
lines.append("=== QUERIES (classify each) ===")
truth = []
for q, i in enumerate(query_idx):
    lines.append(f"Query {q}:\n{verbalize(facts(Gs[i]), focus='structure')}\n")
    truth.append([q, LET[ys[i]]])
open(f"{OUT}/imdb_verbal.txt", 'w').write("\n".join(lines))
json.dump({'imdb': truth}, open(f"{OUT}/imdb_truth.json", 'w'))

print(f"n_shots={len(shot_idx)}  n_queries={len(query_idx)}  chance=0.500")
print(f"(1) logreg / graphlex features   : {acc_feat:.3f}")
print(f"(2) logreg / KumoRFM embeddings  : {acc_kumo:.3f}")
print(f"(3) verbalize + Claude-ICL       : [run subagent on imdb_verbal.txt]")
print(f"\nreference (full-train, our table): networkstats=0.716  kumorfm=0.682")
print(f"wrote {OUT}/imdb_verbal.txt ({len(open(f'{OUT}/imdb_verbal.txt').read())} chars)")
