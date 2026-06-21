"""Cross-domain graph classification: graphlex+LLM vs the specialists.

Same real graphs, same low-label splits, same budget for every method:
  graphlex+LLM : in-context reasoning over verbalize(facts(G))   [run as subagents]
  classical    : logreg on the facts(G) feature vector            [same info as LLM]
  graphpfn/gmn/kumorfm : logreg on precomputed FM embeddings (real graphs)
  majority     : majority-class

The classical arm and the LLM arm consume the IDENTICAL graphlex facts -- one as a
vector, one as prose -- so LLM ~= classical isolates "can the LLM reason over the
verbalized structure as well as a trained linear model?". FM embeddings are the
specialist bar. Emits per-seed prompts + a manifest with truth + baseline accs.

Run: cd /home/scratch/fmsn-dev && source .venv/bin/activate && \
     PYTHONPATH=/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex \
     python /home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex/eval/crossdomain_graphcls.py
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
from graphlex import facts, verbalize, feature_vector, feature_names, SCALAR_GROUPS

TU_ROOT = '/home/scratch/tudata'
EMB_DIR = '/home/scratch/real_fm_embeddings'
OUT = '/home/scratch/bench_out/crossdom_v3'   # v3: + rings + typed-edge (bond) composition
SEEDS = [11, 22, 33]
SHOTS_PER_CLASS = 12
N_QUERY = 40

# (TUDataset name, domain, embedding-key, available FM encoders)
DATASETS = [
    ('IMDB-BINARY', 'social',    'imdb',     ['graphpfn', 'gmn', 'kumorfm']),
    ('PROTEINS',    'biology',   'proteins', ['graphpfn', 'gmn']),
    ('NCI1',        'chemistry', 'nci1',     ['graphpfn', 'gmn']),
]

# Canonical feature set = graphlex A-K scalar groups (single source of truth).
# (Previously a 14-key subset that dropped n_cycles; now unified with the library.)
FKEYS = feature_names()


def node_categories(data):
    """Argmax category per node for one-hot categorical node features, else None."""
    x = data.x
    if x is None:
        return None, 0
    import numpy as _np
    xs = x.numpy()
    rowsums = xs.sum(1)
    if not (_np.allclose(rowsums, 1) and set(_np.unique(xs).tolist()) <= {0.0, 1.0}):
        return None, 0            # not clean one-hot; skip (keep structure-only)
    return xs.argmax(1), xs.shape[1]


def to_nx(data, cats=None):
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    if cats is not None:
        nx.set_node_attributes(G, {i: f"t{int(cats[i])}" for i in G.nodes()}, "type")
    return G


def composition_vec(cats, ncat):
    """Fixed-length category-fraction vector (bag of node types) for the classical arm."""
    if cats is None or ncat == 0:
        return []
    v = np.bincount(cats, minlength=ncat).astype(float)
    return (v / v.sum()).tolist() if v.sum() else v.tolist()


def feat_vec(f, groups=SCALAR_GROUPS):
    return feature_vector(f, groups)


def load(name):
    ds = TUDataset(TU_ROOT, name=name)
    idx = [i for i in range(len(ds))
           if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx])
    return ds, idx, y


def logreg_acc(Xtr, ytr, Xte, yte):
    Xtr = np.nan_to_num(Xtr.astype(np.float64))
    Xte = np.nan_to_num(Xte.astype(np.float64))
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xte)) == yte).mean())


def run_dataset(name, domain, key, encoders):
    ds, idx, y = load(name)
    classes = sorted(set(y.tolist()))
    # precompute graphlex facts feature vectors for ALL kept graphs is expensive;
    # we compute lazily only for the graphs we use (shots+queries) per seed.
    fm = {}
    for enc in encoders:
        p = f"{EMB_DIR}/{enc}__{key}.npz"
        if os.path.exists(p):
            arr = np.load(p)['pooled']
            if arr.shape[0] == len(idx):
                fm[enc] = arr
            else:
                fm[enc] = ('MISMATCH', arr.shape[0], len(idx))
    os.makedirs(f"{OUT}/{name}/ans/opus", exist_ok=True)

    per_seed_base = {enc: [] for enc in encoders}
    per_seed_base['classical'] = []
    per_seed_base['majority'] = []
    files = {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        # stratified shot + query selection over positions in idx
        pos_by_c = {c: [j for j in range(len(idx)) if y[j] == c] for c in classes}
        shot_pos, q_pos = [], []
        for c in classes:
            pc = list(rng.permutation(pos_by_c[c]))
            shot_pos += pc[:SHOTS_PER_CLASS]
            q_pos += pc[SHOTS_PER_CLASS:]
        rng.shuffle(q_pos)
        q_pos = q_pos[:N_QUERY]
        rng.shuffle(shot_pos)

        # graphlex facts/verbalize for selected graphs (with node features if present)
        used = set(shot_pos + q_pos)
        cats_cache, ncat = {}, 0
        for j in used:
            c, nc = node_categories(ds[idx[j]])
            cats_cache[j] = c
            ncat = max(ncat, nc)
        has_attr = any(cats_cache[j] is not None for j in used)
        node_attrs = ['type'] if has_attr else []
        fcache = {j: facts(to_nx(ds[idx[j]], cats_cache[j]), node_attrs=node_attrs)
                  for j in used}

        def fullfeat(j):
            return feat_vec(fcache[j]) + composition_vec(cats_cache[j], ncat)

        # ---- baselines at this split ----
        Xs = np.array([fullfeat(j) for j in shot_pos])
        ys = y[shot_pos]
        Xq = np.array([fullfeat(j) for j in q_pos])
        yq = y[q_pos]
        per_seed_base['classical'].append(logreg_acc(Xs, ys, Xq, yq))
        maj = np.bincount(ys).argmax()
        per_seed_base['majority'].append(float((yq == maj).mean()))
        for enc in encoders:
            if isinstance(fm.get(enc), np.ndarray):
                per_seed_base[enc].append(
                    logreg_acc(fm[enc][shot_pos], ys, fm[enc][q_pos], yq))

        # ---- graphlex+LLM prompt ----
        cls_name = {c: f"CLASS{c}" for c in classes}
        TASK = (f"Each item is a network from the same domain; classify it into one of "
                f"{len(classes)} classes: {', '.join(cls_name[c] for c in classes)}. "
                f"The classes differ in structural properties. Learn the pattern from the "
                f"labeled examples, then classify each query.\nOUTPUT FORMAT: one line per "
                f"query, exactly '<id> <CLASS>' (e.g. '0 {cls_name[classes[0]]}'). No other text.")
        vfocus = 'all' if has_attr else 'structure'
        L = [TASK, "", "=== LABELED EXAMPLES ==="]
        for j in shot_pos:
            L.append(f"[{cls_name[y[j]]}]\n{verbalize(fcache[j], focus=vfocus)}\n")
        L.append("=== QUERIES (classify each) ===")
        truth = []
        for qi, j in enumerate(q_pos):
            L.append(f"Query {qi}:\n{verbalize(fcache[j], focus=vfocus)}\n")
            truth.append([qi, cls_name[int(y[j])]])
        fn = f"{name}/seed{seed}.txt"
        open(f"{OUT}/{fn}", 'w').write("\n".join(L))
        files[fn] = {"dataset": name, "domain": domain, "seed": seed, "truth": truth}

    summary = {k: [float(np.mean(v)), float(np.std(v))] for k, v in per_seed_base.items() if v}
    return files, summary, {e: (fm[e] if not isinstance(fm.get(e), np.ndarray) else 'ok')
                            for e in encoders}, len(idx), classes


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    man = {"seeds": SEEDS, "shots_per_class": SHOTS_PER_CLASS, "n_query": N_QUERY,
           "files": {}, "baselines": {}, "n_graphs": {}}
    for name, domain, key, encoders in DATASETS:
        files, summary, fmstat, ng, classes = run_dataset(name, domain, key, encoders)
        man["files"].update(files)
        man["baselines"][name] = summary
        man["n_graphs"][name] = ng
        print(f"\n=== {name} ({domain}) | {ng} graphs, classes={classes}, "
              f"chance={1/len(classes):.3f} | shots={SHOTS_PER_CLASS}/class, q={N_QUERY} ===")
        print(f"  FM embed status: {fmstat}")
        for k, (mu, sd) in sorted(summary.items(), key=lambda kv: -kv[1][0]):
            print(f"  {k:12} {mu:.3f} +/- {sd:.3f}")
    json.dump(man, open(f"{OUT}/manifest.json", 'w'), indent=0)
    print(f"\nmanifest -> {OUT}/manifest.json  (LLM arm: run subagents on the seed*.txt files)")
