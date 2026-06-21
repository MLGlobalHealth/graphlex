"""Zero-label capability: the task type classical+logreg structurally CANNOT do.

graphlex+LLM reads verbalize(facts(G)) and answers with 0 (or few) labeled
examples, using the model's network-science / chemistry priors. logreg needs
labels to train -> at 0 labels it is undefined (chance). We report the
label-efficiency curve so the capability gap (0-label) and the eventual
convergence (many labels, where logreg catches up) are both visible.

Honest framing (per CROSSDOMAIN_PLAN.md leakage warning): the point is NOT that
the LLM beats a trained logreg (features can be near-sufficient) -- it is that the
LLM works at ZERO/one labels where logreg cannot be trained at all.

Two domains:
  family : ER/BA/WS structural family (network-science prior). graphlex structure.
  mutag  : MUTAG mutagenicity (chemistry prior). graphlex all (rings+bonds+elements).

LLM prompts emitted at shot in {0,3}/class; logreg computed at k in {0,1,3,10,all}.
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
from graphlex import facts, verbalize, feature_vector, feature_names, SCALAR_GROUPS

OUT = '/home/scratch/bench_out/zerolabel'
SEEDS = [11, 22, 33]
LLM_SHOTS = [0, 3]                 # per class, in-context
LOGREG_K = [1, 3, 10]             # per class (k=0 -> chance, reported separately)
NQ = 30
# Canonical feature set = graphlex A-K scalar groups (single source of truth).
FKEYS = feature_names()


def fvec(f, groups=SCALAR_GROUPS):
    return feature_vector(f, groups)


# ---------- family (synthetic) ----------
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
    fams = ['ER', 'BA', 'WS']
    name = {'ER': 'RANDOM (Erdos-Renyi)', 'BA': 'SCALE-FREE (preferential attachment)',
            'WS': 'SMALL-WORLD (Watts-Strogatz)'}
    letter = {'ER': 'R', 'BA': 'S', 'WS': 'W'}
    pool = {f: [gen_family(f, rng) for _ in range(12)] for f in fams}
    queries = []
    for f in fams:
        for G in pool[f][-NQ // 3:]:
            queries.append((G, letter[f]))
    rng.shuffle(queries)
    task = ("Classify each network into one of three structural families:\n"
            f"  R = {name['ER']}\n  S = {name['BA']}\n  W = {name['WS']}\n"
            "OUTPUT FORMAT: one line per query, exactly '<id> <LETTER>' (e.g. '3 S'). No other text.")
    shotpool = {letter[f]: pool[f][:6] for f in fams}
    desc = lambda G: verbalize(facts(G), focus='structure')
    return task, shotpool, queries, desc, ['R', 'S', 'W']


# ---------- mutag (real chemistry) ----------
def mutag_data(seed):
    from torch_geometric.datasets import TUDataset
    from torch_geometric.utils import to_networkx
    ATOMS = {0: 'Carbon', 1: 'Nitrogen', 2: 'Oxygen', 3: 'Fluorine', 4: 'Iodine',
             5: 'Chlorine', 6: 'Bromine'}
    ds = TUDataset('/home/scratch/tudata', name='MUTAG')
    idx = [i for i in range(len(ds)) if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx])
    rng = np.random.RandomState(seed)

    def G_of(j):
        G = to_networkx(ds[idx[j]], to_undirected=True)
        G.remove_edges_from(nx.selfloop_edges(G))
        cats = ds[idx[j]].x.argmax(1).numpy()
        nx.set_node_attributes(G, {i: ATOMS[int(cats[i])] for i in G.nodes()}, 'atom')
        return G
    classes = [0, 1]
    lab = {0: 'NONMUTAGENIC', 1: 'MUTAGENIC'}
    pos = {c: list(rng.permutation([j for j in range(len(idx)) if y[j] == c])) for c in classes}
    shotpool = {lab[c]: [G_of(j) for j in pos[c][:6]] for c in classes}
    qpool = [j for c in classes for j in pos[c][6:6 + NQ // 2 + 5]]
    rng.shuffle(qpool); qpool = qpool[:NQ]
    queries = [(G_of(j), lab[int(y[j])]) for j in qpool]
    task = ("Each item is a molecule (atoms, bonds, rings). Predict whether it is "
            "MUTAGENIC or NONMUTAGENIC. Use chemistry knowledge (e.g. aromatic rings, "
            "nitro N-O groups raise mutagenicity).\nOUTPUT FORMAT: one line per query, "
            "exactly '<id> <LABEL>' where LABEL is MUTAGENIC or NONMUTAGENIC. No other text.")
    desc = lambda G: verbalize(facts(G, node_attrs=['atom']), focus='all')
    return task, shotpool, queries, desc, ['MUTAGENIC', 'NONMUTAGENIC']


def build_prompt(task, shotpool, queries, desc, shot):
    L = [task, ""]
    if shot > 0:
        L.append("=== LABELED EXAMPLES ===")
        ex = []
        for labv, gs in shotpool.items():
            for G in gs[:shot]:
                ex.append((labv, G))
        for labv, G in ex:
            L.append(f"[{labv}]\n{desc(G)}\n")
    L.append("=== QUERIES (classify each) ===")
    truth = []
    for qi, (G, labv) in enumerate(queries):
        L.append(f"Query {qi}:\n{desc(G)}\n")
        truth.append([qi, labv])
    return "\n".join(L), truth


def logreg_curve(shotpool, queries, desc_facts, labels):
    """logreg accuracy at k labels/class. Uses facts() feature vectors."""
    Xq = np.array([desc_facts(G) for G, _ in queries])
    yq = np.array([lab for _, lab in queries])
    out = {}
    for k in LOGREG_K:
        Xtr, ytr = [], []
        for labv, gs in shotpool.items():
            for G in gs[:k]:
                Xtr.append(desc_facts(G)); ytr.append(labv)
        Xtr = np.nan_to_num(np.array(Xtr)); ytr = np.array(ytr)
        if len(set(ytr.tolist())) < 2:
            out[k] = None; continue
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
        out[k] = float((clf.predict(sc.transform(np.nan_to_num(Xq))) == yq).mean())
    return out


if __name__ == "__main__":
    man = {"tasks": {}, "llm_shots": LLM_SHOTS, "logreg_k": LOGREG_K}
    for tname, loader in [('family', family_data), ('mutag', mutag_data)]:
        os.makedirs(f"{OUT}/{tname}/ans/opus", exist_ok=True)
        files = {}
        lrc = {str(k): [] for k in LOGREG_K}
        chances = []
        for seed in SEEDS:
            task, shotpool, queries, desc, labels = loader(seed)
            chances.append(1.0 / len(labels))
            dfacts = (lambda G: fvec(facts(G))) if tname == 'family' \
                else (lambda G: fvec(facts(G, node_attrs=['atom'])))
            lc = logreg_curve(shotpool, queries, dfacts, labels)
            for k, v in lc.items():
                lrc[str(k)].append(v)
            for shot in LLM_SHOTS:
                prompt, truth = build_prompt(task, shotpool, queries, desc, shot)
                fn = f"{tname}/seed{seed}_shot{shot}.txt"
                open(f"{OUT}/{fn}", 'w').write(prompt)
                files[fn] = {"task": tname, "seed": seed, "shot": shot, "truth": truth}
        man["tasks"][tname] = {"files": files,
                               "logreg": {k: [x for x in v if x is not None] for k, v in lrc.items()},
                               "chance": float(np.mean(chances))}
        print(f"[{tname}] chance={np.mean(chances):.3f}  logreg curve:")
        for k in LOGREG_K:
            vs = [x for x in lrc[str(k)] if x is not None]
            if vs:
                print(f"    logreg @ {k:>2}/class : {np.mean(vs):.3f} +/- {np.std(vs):.3f}")
    json.dump(man, open(f"{OUT}/manifest.json", 'w'), indent=0)
    print(f"\nwrote prompts to {OUT}. LLM arm: run subagents on each seed*_shot*.txt")
