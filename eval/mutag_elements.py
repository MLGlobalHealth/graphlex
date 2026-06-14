"""Does giving the LLM REAL element names help vs opaque type ids?

MUTAG is the one chemistry TUDataset with a canonical published node-label ->
atom mapping (Debnath et al. 1991, standard in the graph-kernel literature):
  0=Carbon 1=Nitrogen 2=Oxygen 3=Fluorine 4=Iodine 5=Chlorine 6=Bromine
(NCI1 ships integer labels with NO legend, so we cannot name its atoms.)

Two graphlex+LLM arms, identical graphs/splits:
  opaque : node composition rendered as 't0 70%, t1 10%, ...'
  elem   : node composition rendered as 'Carbon 70%, Nitrogen 10%, ...'
Plus classical (logreg on facts + composition fractions) and majority.
Task: predict mutagenicity (binary).

Run: cd /home/scratch/fmsn-dev && source .venv/bin/activate && \
 PYTHONPATH=/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex \
 python /home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex/eval/mutag_elements.py
"""
import os, json
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import sys
sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
from graphlex import facts, verbalize

OUT = '/home/scratch/bench_out/mutag_elem2'
SEEDS = [11, 22, 33]
SHOTS_PER_CLASS = 10
N_QUERY = 40
ATOMS = {0: 'Carbon', 1: 'Nitrogen', 2: 'Oxygen', 3: 'Fluorine',
         4: 'Iodine', 5: 'Chlorine', 6: 'Bromine'}
FKEYS = ['n_nodes', 'n_edges', 'density', 'n_components', 'mean_degree', 'max_degree',
         'degree_std', 'max_over_mean_degree', 'avg_clustering', 'transitivity',
         'degree_assortativity', 'avg_path_length', 'diameter', 'n_communities']


def feat_vec(f):
    s = f['structure']
    return [0.0 if (s[k] is None or (isinstance(s[k], float) and s[k] != s[k])) else float(s[k])
            for k in FKEYS]


def to_nx(d, cats, names):
    G = to_networkx(d, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    nx.set_node_attributes(G, {i: names[int(cats[i])] for i in G.nodes()}, 'atom')
    return G


if __name__ == "__main__":
    os.makedirs(f"{OUT}/ans/opus", exist_ok=True)
    ds = TUDataset('/home/scratch/tudata', name='MUTAG')
    idx = [i for i in range(len(ds)) if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx])
    classes = sorted(set(y.tolist()))
    # sanity: most-frequent atom should be Carbon (label 0)
    allcats = np.concatenate([ds[i].x.argmax(1).numpy() for i in idx])
    vals, cnts = np.unique(allcats, return_counts=True)
    print("MUTAG atom-label distribution:",
          {ATOMS[int(v)]: int(c) for v, c in sorted(zip(vals, cnts), key=lambda t: -t[1])})

    OPAQUE = {k: f"t{k}" for k in ATOMS}
    man = {"files": {}, "classical": [], "majority": [], "chance": 1 / len(classes)}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        pos = {c: [j for j in range(len(idx)) if y[j] == c] for c in classes}
        sh, q = [], []
        for c in classes:
            pc = list(rng.permutation(pos[c])); sh += pc[:SHOTS_PER_CLASS]; q += pc[SHOTS_PER_CLASS:]
        rng.shuffle(q); q = q[:N_QUERY]; rng.shuffle(sh)
        cats = {j: ds[idx[j]].x.argmax(1).numpy() for j in set(sh + q)}
        ncat = max(ATOMS) + 1

        # classical (facts + atom composition fractions), naming-invariant
        def fc(j, names):
            return facts(to_nx(ds[idx[j]], cats[j], names), node_attrs=['atom'])
        def comp(j):
            v = np.bincount(cats[j], minlength=ncat).astype(float); return (v / v.sum()).tolist()
        Xs = np.array([feat_vec(fc(j, ATOMS)) + comp(j) for j in sh]); ys = y[sh]
        Xq = np.array([feat_vec(fc(j, ATOMS)) + comp(j) for j in q]); yq = y[q]
        scaler = StandardScaler().fit(np.nan_to_num(Xs))
        clf = LogisticRegression(max_iter=5000).fit(scaler.transform(np.nan_to_num(Xs)), ys)
        man["classical"].append(float((clf.predict(scaler.transform(np.nan_to_num(Xq))) == yq).mean()))
        man["majority"].append(float((yq == np.bincount(ys).argmax()).mean()))

        cls = {c: f"CLASS{c}" for c in classes}
        TASK = (f"Each item is a molecule (graph of atoms). Classify it into "
                f"{', '.join(cls[c] for c in classes)} (mutagenic vs not). Learn from the "
                f"labeled examples, then classify each query.\nOUTPUT FORMAT: one line per "
                f"query, exactly '<id> <CLASS>' (e.g. '0 {cls[classes[0]]}'). No other text.")
        for arm, names in [('opaque', OPAQUE), ('elem', ATOMS)]:
            fcache = {j: fc(j, names) for j in set(sh + q)}
            L = [TASK, "", "=== LABELED EXAMPLES ==="]
            for j in sh:
                L.append(f"[{cls[y[j]]}]\n{verbalize(fcache[j], focus='all')}\n")
            L.append("=== QUERIES (classify each) ===")
            truth = []
            for qi, j in enumerate(q):
                L.append(f"Query {qi}:\n{verbalize(fcache[j], focus='all')}\n")
                truth.append([qi, cls[int(y[j])]])
            fn = f"seed{seed}_{arm}.txt"
            open(f"{OUT}/{fn}", 'w').write("\n".join(L))
            man["files"][fn] = {"arm": arm, "seed": seed, "truth": truth}

    json.dump(man, open(f"{OUT}/manifest.json", 'w'), indent=0)
    print(f"classical (facts+composition): {np.mean(man['classical']):.3f} +/- {np.std(man['classical']):.3f}")
    print(f"majority                     : {np.mean(man['majority']):.3f} +/- {np.std(man['majority']):.3f}")
    print(f"chance={man['chance']:.3f}; wrote prompts to {OUT} (run subagents on opaque vs elem)")
