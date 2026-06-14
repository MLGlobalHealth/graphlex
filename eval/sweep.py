"""Broad cross-domain sweep: graphlex+LLM vs classical+logreg vs majority across
many real datasets spanning many sciences. Low-label budget (the LLM's regime).

Reads /home/scratch/bench_out/probe_datasets.json (the working datasets), and for
each: graphlex facts/verbalize (structure + node-attr composition where present),
stratified shots+queries per seed, logreg-on-facts + majority baselines, and a
paste-ready LLM prompt. Drive the LLM arm with run_qwen.py (Qwen, all) and Opus
subagents (subset). Score with score_sweep.py.

Run: cd /home/scratch/fmsn-dev && source .venv/bin/activate && \
 PYTHONPATH=<graphlex> python eval/sweep.py
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

OUT = '/home/scratch/bench_out/sweep'
PROBE = '/home/scratch/bench_out/probe_datasets.json'
TU_ROOT = '/home/scratch/tudata'
SEEDS = [11, 22, 33]
SHOTS_PER_CLASS = 5
MAX_SHOTS = 60          # cap total in-context shots (many-class datasets)
NQ = 40
SKIP_HUGE_AVGN = 150    # skip datasets whose graphs are enormous (cost); reported
POOL_CAP = 4000         # only consider first N graphs (avoid iterating 100k-graph sets)
FKEYS = ['n_nodes', 'n_edges', 'density', 'n_components', 'mean_degree', 'max_degree',
         'degree_std', 'max_over_mean_degree', 'avg_clustering', 'transitivity',
         'degree_assortativity', 'avg_path_length', 'diameter', 'n_cycles', 'n_communities']


def fvec(f):
    s = f['structure']
    return [0.0 if (s[k] is None or (isinstance(s[k], float) and s[k] != s[k])) else float(s[k])
            for k in FKEYS]


def node_cats(data):
    x = data.x
    if x is None:
        return None, 0
    xs = x.numpy()
    if not (np.allclose(xs.sum(1), 1) and set(np.unique(xs).tolist()) <= {0.0, 1.0}):
        return None, 0
    return xs.argmax(1), xs.shape[1]


def to_nx(data, cats):
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    if cats is not None:
        nx.set_node_attributes(G, {i: f"t{int(cats[i])}" for i in G.nodes()}, 'type')
    return G


def comp(cats, ncat):
    if cats is None or ncat == 0:
        return []
    v = np.bincount(cats, minlength=ncat).astype(float)
    return (v / v.sum()).tolist() if v.sum() else v.tolist()


def run_one(name, domain):
    ds = TUDataset(TU_ROOT, name=name)
    cap = min(len(ds), POOL_CAP)
    idx = [i for i in range(cap) if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx])
    classes = sorted(set(y.tolist()))
    spc = max(2, min(SHOTS_PER_CLASS, MAX_SHOTS // len(classes)))
    os.makedirs(f"{OUT}/{name}/ans/opus", exist_ok=True)
    base = {'classical': [], 'majority': []}
    files = {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        pos = {c: list(rng.permutation([j for j in range(len(idx)) if y[j] == c])) for c in classes}
        shot, q = [], []
        for c in classes:
            shot += pos[c][:spc]; q += pos[c][spc:]
        rng.shuffle(q); q = q[:NQ]; rng.shuffle(shot)
        used = set(shot + q)
        cc = {j: node_cats(ds[idx[j]]) for j in used}
        ncat = max((cc[j][1] for j in used), default=0)
        has_attr = any(cc[j][0] is not None for j in used)
        na = ['type'] if has_attr else []
        fcache = {j: facts(to_nx(ds[idx[j]], cc[j][0]), node_attrs=na) for j in used}

        def full(j):
            return fvec(fcache[j]) + comp(cc[j][0], ncat)
        Xs = np.array([full(j) for j in shot]); ys = y[shot]
        Xq = np.array([full(j) for j in q]); yq = y[q]
        sc = StandardScaler().fit(np.nan_to_num(Xs))
        clf = LogisticRegression(max_iter=4000).fit(sc.transform(np.nan_to_num(Xs)), ys)
        base['classical'].append(float((clf.predict(sc.transform(np.nan_to_num(Xq))) == yq).mean()))
        base['majority'].append(float((yq == np.bincount(ys).argmax()).mean()))

        clsname = {c: f"CLASS{c}" for c in classes}
        focus = 'all' if has_attr else 'structure'
        TASK = (f"Each item is a network from the '{domain}' domain ({name}); classify it "
                f"into one of {len(classes)}: {', '.join(clsname[c] for c in classes)}. "
                f"Learn the pattern from the labeled examples, then classify each query.\n"
                f"OUTPUT FORMAT: one line per query, exactly '<id> <CLASS>'. No other text.")
        L = [TASK, "", "=== LABELED EXAMPLES ==="]
        for j in shot:
            L.append(f"[{clsname[y[j]]}]\n{verbalize(fcache[j], focus=focus)}\n")
        L.append("=== QUERIES (classify each) ===")
        truth = []
        for qi, j in enumerate(q):
            L.append(f"Query {qi}:\n{verbalize(fcache[j], focus=focus)}\n")
            truth.append([qi, clsname[int(y[j])]])
        fn = f"{name}/seed{seed}.txt"
        open(f"{OUT}/{fn}", 'w').write("\n".join(L))
        files[fn] = {"dataset": name, "domain": domain, "seed": seed, "truth": truth,
                     "shots_per_class": spc, "n_classes": len(classes)}
    return files, {k: [float(np.mean(v)), float(np.std(v))] for k, v in base.items()}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    probe = json.load(open(PROBE))
    man = {"seeds": SEEDS, "files": {}, "baselines": {}, "meta": {}}
    for d in probe:
        if d['avgN'] > SKIP_HUGE_AVGN:
            print(f"skip {d['name']} (avgN {d['avgN']} > {SKIP_HUGE_AVGN})", flush=True)
            continue
        try:
            files, base = run_one(d['name'], d['domain'])
            man['files'].update(files)
            man['baselines'][d['name']] = base
            man['meta'][d['name']] = {'domain': d['domain'], 'chance': d['chance'],
                                      'classes': d['classes']}
            print(f"OK {d['name']:18} {d['domain']:12} classical={base['classical'][0]:.3f} "
                  f"majority={base['majority'][0]:.3f} chance={d['chance']:.3f}", flush=True)
        except Exception as e:
            print(f"ERR {d['name']}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    json.dump(man, open(f"{OUT}/manifest.json", 'w'), indent=0)
    print(f"\n{len(man['baselines'])} datasets -> {OUT}/manifest.json")
