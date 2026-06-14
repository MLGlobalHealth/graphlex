"""Build the 'verbalize the structure' evaluation: synthetic graph-family task
(random / scale-free / small-world). Emit two paste-ready prompts (verbalized
features vs raw edge list), each with K-shot context + M numbered queries asking
for strict parseable predictions. Also compute the logreg-on-features reference.
Ground truth saved for scoring after subagents answer.
"""
import sys, os, json, numpy as np, networkx as nx
sys.path.insert(0, 'src')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RNG = np.random.RandomState(7)
OUT = '/home/scratch/bench_out/verbalize_prompts'; os.makedirs(OUT, exist_ok=True)
KSHOT = 6      # labeled examples per class in context
NQ = 24        # total queries (8 per class)
LETTER = {'ER': 'R', 'BA': 'S', 'WS': 'W'}
NAME = {'R': 'RANDOM (Erdos-Renyi)', 'S': 'SCALE-FREE (preferential attachment)', 'W': 'SMALL-WORLD (Watts-Strogatz)'}

def gen(fam):
    n = int(RNG.randint(20, 41))
    for _ in range(20):
        if fam == 'ER': G = nx.erdos_renyi_graph(n, RNG.uniform(0.12, 0.25), seed=int(RNG.randint(1e6)))
        elif fam == 'BA': G = nx.barabasi_albert_graph(n, int(RNG.choice([2, 3])), seed=int(RNG.randint(1e6)))
        else: G = nx.watts_strogatz_graph(n, int(RNG.choice([4, 6])), RNG.uniform(0.1, 0.3), seed=int(RNG.randint(1e6)))
        if nx.is_connected(G) and G.number_of_edges() >= 1: return G
    return G

def feats(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()], float)
    Gc = G.subgraph(max(nx.connected_components(G), key=len))
    apl = nx.average_shortest_path_length(Gc) if Gc.number_of_nodes() > 1 else 0
    try: assort = nx.degree_assortativity_coefficient(G)
    except Exception: assort = 0.0
    assort = 0.0 if not np.isfinite(assort) else assort
    return {
        'n': n, 'm': m, 'density': nx.density(G), 'mean_deg': deg.mean(), 'max_deg': int(deg.max()),
        'hub_ratio': deg.max() / max(deg.mean(), 1e-9), 'deg_std': deg.std(),
        'clustering': nx.average_clustering(G), 'transitivity': nx.transitivity(G),
        'avg_path_len': apl, 'diameter': nx.diameter(Gc) if Gc.number_of_nodes() > 1 else 0,
        'assortativity': assort, 'frac_hubs': float((deg > 2 * deg.mean()).mean()),
    }

FKEYS = ['n', 'm', 'density', 'mean_deg', 'max_deg', 'hub_ratio', 'deg_std', 'clustering',
         'transitivity', 'avg_path_len', 'diameter', 'assortativity', 'frac_hubs']

def verbalize(f):
    return (f"- {f['n']} nodes, {f['m']} edges, density {f['density']:.2f}\n"
            f"- mean degree {f['mean_deg']:.1f}, max degree {f['max_deg']} (max/mean ratio {f['hub_ratio']:.1f}, degree std {f['deg_std']:.1f})\n"
            f"- fraction of high-degree hubs {f['frac_hubs']:.2f}\n"
            f"- avg clustering {f['clustering']:.2f}, transitivity {f['transitivity']:.2f}\n"
            f"- avg shortest path length {f['avg_path_len']:.2f}, diameter {f['diameter']}\n"
            f"- degree assortativity {f['assortativity']:.2f}")

def edgelist(G):
    return "edges: " + ", ".join(f"({a},{b})" for a, b in sorted(tuple(sorted(e)) for e in G.edges()))

def edgelist_perm(G):
    # randomly relabel nodes -> destroys generator index artifacts (ring lattice / low-index hubs)
    nodes = list(G.nodes()); perm = list(RNG.permutation(len(nodes)))
    mp = {nodes[i]: perm[i] for i in range(len(nodes))}
    H = nx.relabel_nodes(G, mp)
    return "edges: " + ", ".join(f"({a},{b})" for a, b in sorted(tuple(sorted(e)) for e in H.edges()))

# build dataset
data = []
for fam in ['ER', 'BA', 'WS']:
    for _ in range(KSHOT + NQ // 3 + 2):
        G = gen(fam); data.append((G, LETTER[fam], feats(G)))
shots, queries = [], []
by = {'R': [], 'S': [], 'W': []}
for rec in data: by[rec[1]].append(rec)
for lab in by:
    shots += [(r, lab) for r in by[lab][:KSHOT]]
    queries += [(r, lab) for r in by[lab][KSHOT:KSHOT + NQ // 3]]
RNG.shuffle(queries)

TASK = ("Classify each small network into one of three structural families:\n"
        f"  R = {NAME['R']}\n  S = {NAME['S']}\n  W = {NAME['W']}\n"
        "Learn the pattern from the labeled examples, then classify each query.\n"
        "OUTPUT FORMAT: one line per query, exactly '<id> <LETTER>' (e.g. '3 S'). No other text.")

def build(arm):
    desc = {'verbal': (lambda G, f: verbalize(f)),
            'raw': (lambda G, f: edgelist(G)),
            'raw_perm': (lambda G, f: edgelist_perm(G))}[arm]
    L = [TASK, "", "=== LABELED EXAMPLES ==="]
    for (rec, lab) in shots:
        G, _, f = rec; L.append(f"[{NAME[lab][0]}={lab}]\n{desc(G, f)}\n")
    L.append("=== QUERIES (classify each) ===")
    truth = []
    for i, (rec, lab) in enumerate(queries):
        G, _, f = rec; L.append(f"Query {i}:\n{desc(G, f)}\n"); truth.append([i, lab])
    return "\n".join(L), truth

truth = None
for arm in ['verbal', 'raw', 'raw_perm']:
    prompt, truth = build(arm)
    open(f"{OUT}/synth_{arm}.txt", 'w').write(prompt)
    print(f"wrote synth_{arm}.txt  ({len(prompt)} chars)")
json.dump({'synth': truth}, open(f"{OUT}/truth.json", 'w'))

# logreg reference on the SAME features/split
Xs = np.array([[s[0][2][k] for k in FKEYS] for s in shots]); ys = np.array([s[1] for s in shots])
Xq = np.array([[q[0][2][k] for k in FKEYS] for q in queries]); yq = np.array([q[1] for q in queries])
sc = StandardScaler().fit(Xs)
clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xs), ys)
acc = float((clf.predict(sc.transform(Xq)) == yq).mean())
print(f"\nLOGREG reference (same features, same {len(shots)}-shot context): {acc:.3f}  (chance=0.333, n_query={len(yq)})")
print(f"truth distribution: {dict(zip(*np.unique(yq, return_counts=True)))}")
