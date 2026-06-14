"""Multi-seed synthetic graph-family ID with the controls a top venue demands.

Fixes the two pilot confounds:
  * named-family PRIOR LEAK: adds an ANONYMIZED-label arm (families relabeled
    Class A/B/C, mapping reshuffled per seed) so the LLM cannot use its textbook
    prior about "scale-free"/"small-world" and must learn from the shots.
  * single-run: runs >=5 seeds; emits per-(seed,arm) prompts + a manifest with
    ground truth and the per-seed logreg reference.

Arms (left->right = less->more usable signal):
  raw          raw edge list, original node labels (generator index artifact present)
  raw_perm     raw edge list, node labels permuted (artifact destroyed)  [CONTROL]
  counts_only  minimal computed verbalization (n, m, density, components)  [ladder floor]
  verbal       full computed-feature verbalization, family NAMES in task
  verbal_anon  full computed-feature verbalization, ANON labels A/B/C       [DECISIVE]

Run:  /home/scratch/fmsn-dev/.venv/bin/python eval/synth_multiseed.py
"""
import sys, os, json
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = '/home/scratch/bench_out/synth_v2'
os.makedirs(OUT, exist_ok=True)
SEEDS = [11, 22, 33, 44, 55]
KSHOT = 6          # labeled examples per class
NPC = 10           # queries per class  -> NQ = 30
FAMS = ['ER', 'BA', 'WS']
NAMED = {'ER': 'R', 'BA': 'S', 'WS': 'W'}
NAME = {'R': 'RANDOM (Erdos-Renyi)', 'S': 'SCALE-FREE (preferential attachment)',
        'W': 'SMALL-WORLD (Watts-Strogatz)'}


def gen(fam, rng):
    n = int(rng.randint(20, 41))
    G = None
    for _ in range(40):
        if fam == 'ER':
            G = nx.erdos_renyi_graph(n, rng.uniform(0.12, 0.25), seed=int(rng.randint(1e6)))
        elif fam == 'BA':
            G = nx.barabasi_albert_graph(n, int(rng.choice([2, 3])), seed=int(rng.randint(1e6)))
        else:
            G = nx.watts_strogatz_graph(n, int(rng.choice([4, 6])), rng.uniform(0.1, 0.3),
                                        seed=int(rng.randint(1e6)))
        if nx.is_connected(G) and G.number_of_edges() >= 1:
            return G
    return G


FKEYS = ['n', 'm', 'density', 'mean_deg', 'max_deg', 'hub_ratio', 'deg_std', 'clustering',
         'transitivity', 'avg_path_len', 'diameter', 'assortativity', 'frac_hubs']


def feats(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()], float)
    Gc = G.subgraph(max(nx.connected_components(G), key=len))
    apl = nx.average_shortest_path_length(Gc) if Gc.number_of_nodes() > 1 else 0
    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = 0.0
    assort = 0.0 if not np.isfinite(assort) else assort
    return {'n': n, 'm': m, 'density': nx.density(G), 'mean_deg': deg.mean(),
            'max_deg': int(deg.max()), 'hub_ratio': deg.max() / max(deg.mean(), 1e-9),
            'deg_std': deg.std(), 'clustering': nx.average_clustering(G),
            'transitivity': nx.transitivity(G), 'avg_path_len': apl,
            'diameter': nx.diameter(Gc) if Gc.number_of_nodes() > 1 else 0,
            'assortativity': assort, 'frac_hubs': float((deg > 2 * deg.mean()).mean())}


def verbalize(f):
    return (f"- {f['n']} nodes, {f['m']} edges, density {f['density']:.2f}\n"
            f"- mean degree {f['mean_deg']:.1f}, max degree {f['max_deg']} "
            f"(max/mean ratio {f['hub_ratio']:.1f}, degree std {f['deg_std']:.1f})\n"
            f"- fraction of high-degree hubs {f['frac_hubs']:.2f}\n"
            f"- avg clustering {f['clustering']:.2f}, transitivity {f['transitivity']:.2f}\n"
            f"- avg shortest path length {f['avg_path_len']:.2f}, diameter {f['diameter']}\n"
            f"- degree assortativity {f['assortativity']:.2f}")


def counts_only(f):
    return (f"- {f['n']} nodes, {f['m']} edges, density {f['density']:.2f}\n"
            f"- 1 connected component")


def edgelist(G, rng=None):
    H = G
    if rng is not None:
        nodes = list(G.nodes())
        perm = list(rng.permutation(len(nodes)))
        H = nx.relabel_nodes(G, {nodes[i]: perm[i] for i in range(len(nodes))})
    return "edges: " + ", ".join(f"({a},{b})" for a, b in
                                 sorted(tuple(sorted(e)) for e in H.edges()))


def build_dataset(rng):
    by = {NAMED[fam]: [] for fam in FAMS}
    for fam in FAMS:
        for _ in range(KSHOT + NPC):
            G = gen(fam, rng)
            by[NAMED[fam]].append((G, feats(G)))
    shots, queries = [], []
    for lab in by:
        shots += [(rec, lab) for rec in by[lab][:KSHOT]]
        queries += [(rec, lab) for rec in by[lab][KSHOT:KSHOT + NPC]]
    rng.shuffle(queries)
    return shots, queries


def task_named():
    return ("Classify each small network into one of three structural families:\n"
            f"  R = {NAME['R']}\n  S = {NAME['S']}\n  W = {NAME['W']}\n"
            "Learn the pattern from the labeled examples, then classify each query.\n"
            "OUTPUT FORMAT: one line per query, exactly '<id> <LETTER>' (e.g. '3 S'). "
            "No other text.")


def task_anon():
    return ("Each small network belongs to one of three structural families, called "
            "Class A, Class B, and Class C. The families differ only in their "
            "structural properties (you are NOT told which is which). Learn the "
            "pattern from the labeled examples, then classify each query.\n"
            "OUTPUT FORMAT: one line per query, exactly '<id> <CLASS>' (e.g. '3 B'). "
            "No other text.")


def emit(seed):
    rng = np.random.RandomState(seed)
    shots, queries = build_dataset(rng)
    # per-seed anonymization map  R/S/W -> A/B/C (reshuffled each seed)
    anon = dict(zip(['R', 'S', 'W'], list(rng.permutation(['A', 'B', 'C']))))

    arms = {
        'raw':         (task_named(), lambda G, f: edgelist(G), False),
        'raw_perm':    (task_named(), lambda G, f: edgelist(G, rng), False),
        'counts_only': (task_named(), lambda G, f: counts_only(f), False),
        'verbal':      (task_named(), lambda G, f: verbalize(f), False),
        'verbal_anon': (task_anon(),  lambda G, f: verbalize(f), True),
    }
    manifest = {}
    for arm, (task, desc, is_anon) in arms.items():
        L = [task, "", "=== LABELED EXAMPLES ==="]
        for (rec, lab) in shots:
            G, f = rec
            shown = anon[lab] if is_anon else lab
            tag = f"Class {shown}" if is_anon else NAME[lab][0] + "=" + lab
            L.append(f"[{tag}]\n{desc(G, f)}\n")
        L.append("=== QUERIES (classify each) ===")
        truth = []
        for i, (rec, lab) in enumerate(queries):
            G, f = rec
            L.append(f"Query {i}:\n{desc(G, f)}\n")
            truth.append([i, (anon[lab] if is_anon else lab)])
        fn = f"seed{seed}_{arm}.txt"
        open(f"{OUT}/{fn}", 'w').write("\n".join(L))
        manifest[fn] = {"arm": arm, "seed": seed, "truth": truth,
                        "labels": sorted(set(t[1] for t in truth))}

    # logreg reference (same features, same split) -- naming-invariant
    Xs = np.array([[s[0][1][k] for k in FKEYS] for s in shots])
    ys = np.array([s[1] for s in shots])
    Xq = np.array([[q[0][1][k] for k in FKEYS] for q in queries])
    yq = np.array([q[1] for q in queries])
    sc = StandardScaler().fit(Xs)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xs), ys)
    acc = float((clf.predict(sc.transform(Xq)) == yq).mean())
    return manifest, acc, len(queries)


if __name__ == "__main__":
    full = {"seeds": SEEDS, "chance": 1 / 3, "files": {}, "logreg": {}, "nq": None}
    for s in SEEDS:
        m, acc, nq = emit(s)
        full["files"].update(m)
        full["logreg"][str(s)] = acc
        full["nq"] = nq
        print(f"seed {s}: wrote 5 arms, {nq} queries; logreg ref = {acc:.3f}")
    json.dump(full, open(f"{OUT}/manifest.json", 'w'), indent=0)
    lr = list(full["logreg"].values())
    print(f"\nlogreg reference across {len(SEEDS)} seeds: "
          f"mean {np.mean(lr):.3f} +/- {np.std(lr):.3f}  (chance 0.333, nq={nq}/seed)")
    print(f"manifest -> {OUT}/manifest.json")
