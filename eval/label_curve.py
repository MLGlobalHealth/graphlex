"""Label-efficiency crossover: is the LLM 'strictly worse than logreg if it can run'?

Test at the SAME low label budgets where logreg CAN run (k>=1). Hypothesis: the
LLM's prior makes it more label-efficient, so LLM >= logreg at low k, with a
crossover where logreg overtakes once it has enough labels — and the crossover
depends on how relevant the LLM's priors are to the domain.

Two domains:
  family   : ER/BA/WS structural family — STRONG network-science prior.
  proteins : PROTEINS enzyme/not — REAL bio, classes arbitrary to the LLM (weak prior).

logreg at k in {1,2,3,5,8,12}/class (local). LLM prompts at k in {1,3,5}/class.
Run: cd /home/scratch/fmsn-dev && source .venv/bin/activate && \
 PYTHONPATH=<graphlex> python <this>
"""
import os, json
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
from graphlex import facts, verbalize

OUT = '/home/scratch/bench_out/labelcurve'
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]   # 8 seeds (Qwen runs all; Claude anchors at 11/22/33)
K_LOGREG = [1, 2, 3, 5, 8, 12]
K_LLM = [1, 3, 5]
NQ = 30
POOL = 12   # shots available per class
FKEYS = ['n_nodes', 'n_edges', 'density', 'n_components', 'mean_degree', 'max_degree',
         'degree_std', 'max_over_mean_degree', 'avg_clustering', 'transitivity',
         'degree_assortativity', 'avg_path_length', 'diameter', 'n_cycles', 'n_communities']


def fvec(f):
    s = f['structure']
    return [0.0 if (s[k] is None or (isinstance(s[k], float) and s[k] != s[k])) else float(s[k])
            for k in FKEYS]


def gen_family(fam, rng):
    n = int(rng.randint(20, 41))
    for _ in range(40):
        if fam == 'ER':
            G = nx.erdos_renyi_graph(n, rng.uniform(0.12, 0.25), seed=int(rng.randint(1e6)))
        elif fam == 'BA':
            G = nx.barabasi_albert_graph(n, int(rng.choice([2, 3])), seed=int(rng.randint(1e6)))
        else:
            G = nx.watts_strogatz_graph(n, int(rng.choice([4, 6])), rng.uniform(0.1, 0.3),
                                        seed=int(rng.randint(1e6)))
        if nx.is_connected(G):
            return G
    return G


def family_data(seed):
    rng = np.random.RandomState(seed)
    fams = ['ER', 'BA', 'WS']; let = {'ER': 'R', 'BA': 'S', 'WS': 'W'}
    nm = {'ER': 'RANDOM (Erdos-Renyi)', 'BA': 'SCALE-FREE (preferential attachment)',
          'WS': 'SMALL-WORLD (Watts-Strogatz)'}
    pool = {f: [gen_family(f, rng) for _ in range(POOL)] for f in fams}
    q = [gen_family(f, rng) for f in fams for _ in range(NQ // 3)]
    qlab = [let[f] for f in fams for _ in range(NQ // 3)]
    order = list(rng.permutation(len(q)))
    queries = [(q[i], qlab[i]) for i in order]
    task = ("Classify each network into one of three structural families:\n"
            f"  R = {nm['ER']}\n  S = {nm['BA']}\n  W = {nm['WS']}\n"
            "OUTPUT FORMAT: one line per query, exactly '<id> <LETTER>' (e.g. '3 S'). No other text.")
    shotpool = {let[f]: pool[f] for f in fams}
    desc = lambda G: verbalize(facts(G), focus='structure')
    dfacts = lambda G: fvec(facts(G))
    return task, shotpool, queries, desc, dfacts, ['R', 'S', 'W']


def proteins_data(seed):
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_networkx
    ds = TUDataset('/home/scratch/tudata', name='PROTEINS')
    idx = [i for i in range(len(ds)) if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx]); rng = np.random.RandomState(seed)
    ncat = ds[0].x.shape[1]

    def G_of(j):
        G = to_networkx(ds[idx[j]], to_undirected=True); G.remove_edges_from(nx.selfloop_edges(G))
        cats = ds[idx[j]].x.argmax(1).numpy()
        nx.set_node_attributes(G, {i: f"t{int(cats[i])}" for i in G.nodes()}, 'type')
        return G, cats
    def comp(cats):
        v = np.bincount(cats, minlength=ncat).astype(float); return (v / v.sum()).tolist()
    cls = {0: 'CLASS0', 1: 'CLASS1'}
    pos = {c: list(rng.permutation([j for j in range(len(idx)) if y[j] == c])) for c in [0, 1]}
    shotpool, qpool = {}, []
    for c in [0, 1]:
        shotpool[cls[c]] = [G_of(j) for j in pos[c][:POOL]]
        qpool += pos[c][POOL:POOL + NQ // 2 + 5]
    rng.shuffle(qpool); qpool = qpool[:NQ]
    queries = [(G_of(j), cls[int(y[j])]) for j in qpool]
    task = ("Each item is a protein structure graph; classify into CLASS0 or CLASS1. "
            "Learn the pattern from the labeled examples, then classify each query.\n"
            "OUTPUT FORMAT: one line per query, exactly '<id> <CLASS>'. No other text.")
    desc = lambda Gc: verbalize(facts(Gc[0], node_attrs=['type']), focus='all')
    dfacts = lambda Gc: fvec(facts(Gc[0], node_attrs=['type'])) + comp(Gc[1])
    return task, shotpool, queries, desc, dfacts, ['CLASS0', 'CLASS1']


def imdb_data(seed):
    """IMDB-BINARY: social collaboration graphs, NO node features, classes arbitrary
    to the LLM (weak prior). Structure-only verbalization."""
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_networkx
    ds = TUDataset('/home/scratch/tudata', name='IMDB-BINARY')
    idx = [i for i in range(len(ds)) if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx]); rng = np.random.RandomState(seed)

    def G_of(j):
        G = to_networkx(ds[idx[j]], to_undirected=True); G.remove_edges_from(nx.selfloop_edges(G))
        return G
    cls = {0: 'CLASS0', 1: 'CLASS1'}
    pos = {c: list(rng.permutation([j for j in range(len(idx)) if y[j] == c])) for c in [0, 1]}
    shotpool, qpool = {}, []
    for c in [0, 1]:
        shotpool[cls[c]] = [G_of(j) for j in pos[c][:POOL]]
        qpool += pos[c][POOL:POOL + NQ // 2 + 5]
    rng.shuffle(qpool); qpool = qpool[:NQ]
    queries = [(G_of(j), cls[int(y[j])]) for j in qpool]
    task = ("Each item is a movie-actor collaboration network; classify into CLASS0 or "
            "CLASS1. Learn the pattern from the labeled examples, then classify each query.\n"
            "OUTPUT FORMAT: one line per query, exactly '<id> <CLASS>'. No other text.")
    desc = lambda G: verbalize(facts(G), focus='structure')
    dfacts = lambda G: fvec(facts(G))
    return task, shotpool, queries, desc, dfacts, ['CLASS0', 'CLASS1']


def build_prompt(task, shotpool, queries, desc, k):
    L = [task, "", "=== LABELED EXAMPLES ==="]
    for labv, gs in shotpool.items():
        for G in gs[:k]:
            L.append(f"[{labv}]\n{desc(G)}\n")
    L.append("=== QUERIES (classify each) ===")
    truth = []
    for qi, (G, labv) in enumerate(queries):
        L.append(f"Query {qi}:\n{desc(G)}\n"); truth.append([qi, labv])
    return "\n".join(L), truth


def logreg_at(shotpool, queries, dfacts, k):
    Xtr, ytr = [], []
    for labv, gs in shotpool.items():
        for G in gs[:k]:
            Xtr.append(dfacts(G)); ytr.append(labv)
    Xq = np.nan_to_num(np.array([dfacts(G) for G, _ in queries]))
    yq = np.array([l for _, l in queries])
    Xtr = np.nan_to_num(np.array(Xtr)); ytr = np.array(ytr)
    if len(set(ytr.tolist())) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xq)) == yq).mean())


if __name__ == "__main__":
    man = {"k_logreg": K_LOGREG, "k_llm": K_LLM, "tasks": {}}
    for tname, loader in [('family', family_data), ('proteins', proteins_data),
                          ('imdb', imdb_data)]:
        os.makedirs(f"{OUT}/{tname}/ans/opus", exist_ok=True)
        files = {}; lrc = {str(k): [] for k in K_LOGREG}
        for seed in SEEDS:
            task, shotpool, queries, desc, dfacts, labels = loader(seed)
            for k in K_LOGREG:
                v = logreg_at(shotpool, queries, dfacts, k)
                if v is not None:
                    lrc[str(k)].append(v)
            for k in K_LLM:
                prompt, truth = build_prompt(task, shotpool, queries, desc, k)
                fn = f"{tname}/seed{seed}_k{k}.txt"
                open(f"{OUT}/{fn}", 'w').write(prompt)
                files[fn] = {"task": tname, "seed": seed, "k": k, "truth": truth}
        man["tasks"][tname] = {"files": files,
                               "logreg": {k: v for k, v in lrc.items()},
                               "chance": 1.0 / len(labels)}
        print(f"[{tname}] chance {1.0/len(labels):.3f} logreg:",
              {k: round(float(np.mean(v)), 3) for k, v in lrc.items() if v})
    json.dump(man, open(f"{OUT}/manifest.json", 'w'), indent=0)
    print(f"wrote prompts to {OUT}; run subagents on each seed*_k*.txt")
