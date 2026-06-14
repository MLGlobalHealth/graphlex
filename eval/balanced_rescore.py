"""Rescore the sweep with BALANCED accuracy (macro-averaged per-class recall).

Raw accuracy on imbalanced datasets rewards predicting the majority class — which
the majority baseline does but the LLM can't (it sees balanced 5/5 shots, so it has
no base-rate signal). Balanced accuracy removes that artifact: majority -> chance
(1/n_classes), and every method is judged on real per-class discrimination.

Replays sweep.py's deterministic splits to recompute classical + majority preds,
and reads the existing LLM answer files. No prompt regeneration / no LLM re-runs.
"""
import os, re, sys, json
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import TUDataset
sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex/eval')
from sweep import (fvec, node_cats, to_nx, comp, SEEDS, SHOTS_PER_CLASS, MAX_SHOTS,
                   NQ, POOL_CAP, TU_ROOT, OUT)
from graphlex import facts

man = json.load(open(f"{OUT}/manifest.json"))
LINE = re.compile(r'^\s*(?:query\s*)?(\d+)\s*[:.\)\-]?\s+([A-Za-z0-9_]+)\s*$', re.I)


def parse(p):
    d = {}
    for ln in open(p):
        m = LINE.match(ln.strip())
        if m:
            d[int(m.group(1))] = m.group(2).strip().upper()
    return d


def bal_acc(truth_list, pred_map):
    """macro-averaged recall over true classes; pred_map int->TOKEN(upper)."""
    by = {}
    for i, lab in truth_list:
        by.setdefault(str(lab).upper(), []).append(i)
    recs = []
    for lab, ids in by.items():
        recs.append(np.mean([pred_map.get(i) == lab for i in ids]))
    return float(np.mean(recs)) if recs else None


def splits(name):
    """Reproduce sweep.py per-seed (shot,q) index lists -> yields (seed, shot, q, y, idx, ds)."""
    ds = TUDataset(TU_ROOT, name=name)
    cap = min(len(ds), POOL_CAP)
    idx = [i for i in range(cap) if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx])
    classes = sorted(set(y.tolist()))
    spc = max(2, min(SHOTS_PER_CLASS, MAX_SHOTS // len(classes)))
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        pos = {c: list(rng.permutation([j for j in range(len(idx)) if y[j] == c])) for c in classes}
        shot, q = [], []
        for c in classes:
            shot += pos[c][:spc]; q += pos[c][spc:]
        rng.shuffle(q); q = q[:NQ]; rng.shuffle(shot)
        yield seed, shot, q, y, idx, ds, classes


def classical_majority_balacc(name):
    cl, mj = [], []
    for seed, shot, q, y, idx, ds, classes in splits(name):
        used = set(shot + q)
        cc = {j: node_cats(ds[idx[j]]) for j in used}
        ncat = max((cc[j][1] for j in used), default=0)
        na = ['type'] if any(cc[j][0] is not None for j in used) else []
        fc = {j: facts(to_nx(ds[idx[j]], cc[j][0]), node_attrs=na) for j in used}
        full = lambda j: fvec(fc[j]) + comp(cc[j][0], ncat)
        Xs = np.nan_to_num(np.array([full(j) for j in shot])); ys = y[shot]
        Xq = np.nan_to_num(np.array([full(j) for j in q])); yq = y[q]
        sc = StandardScaler().fit(Xs)
        clf = LogisticRegression(max_iter=4000).fit(sc.transform(Xs), ys)
        pc = clf.predict(sc.transform(Xq))
        majc = np.bincount(ys).argmax()
        tl = [(i, f"CLASS{int(yq[i])}") for i in range(len(yq))]
        cl.append(bal_acc(tl, {i: f"CLASS{int(pc[i])}".upper() for i in range(len(pc))}))
        mj.append(bal_acc(tl, {i: f"CLASS{int(majc)}".upper() for i in range(len(yq))}))
    return np.mean(cl), np.mean(mj)


def llm_balacc(name, model):
    out = []
    for fn, meta in man['files'].items():
        if meta['dataset'] != name:
            continue
        a = f"{OUT}/{name}/ans/{model}/seed{meta['seed']}.ans"
        if not os.path.exists(a):
            continue
        pred = parse(a)
        if pred:
            out.append(bal_acc(meta['truth'], pred))
    return np.mean(out) if out else None


rows = []
for ds, meta in sorted(man['meta'].items(), key=lambda kv: (kv[1]['domain'], kv[0])):
    ncls = meta['classes']
    cl, mj = classical_majority_balacc(ds)
    qw = llm_balacc(ds, 'qwen'); q32 = llm_balacc(ds, 'qwen32'); op = llm_balacc(ds, 'opus')
    rows.append((meta['domain'], ds, 1.0/ncls, cl, mj, qw, q32, op))

hdr = f"{'domain':12} {'dataset':16} {'chance':>7} {'classic':>8} {'Qwen14':>7} {'Qwen32':>7} {'Opus':>7}"
print("BALANCED ACCURACY (macro recall; majority==chance)\n")
print(hdr); print('-'*len(hdr))
for dom, ds, ch, cl, mj, qw, q32, op in rows:
    f = lambda x: f"{x:.3f}" if x is not None else "   -"
    print(f"{dom:12} {ds:16} {ch:7.3f} {cl:8.3f} {f(qw):>7} {f(q32):>7} {f(op):>7}")

for nm, gi in [("Qwen-14b", 5), ("Qwen-32b", 6), ("Opus", 7)]:
    reg = [max(r[3], r[4]) - r[gi] for r in rows if r[gi] is not None]
    if reg:
        reg = np.array(reg)
        beat_cl = [r[gi] - r[3] for r in rows if r[gi] is not None]
        print(f"\n{nm} vs best non-LLM (balanced acc): n={len(reg)} | mean regret {reg.mean():+.3f} | "
              f"worst {reg.max():+.3f} | within .05: {int((reg<=.05).sum())}/{len(reg)} | "
              f">classical: {int((np.array(beat_cl)>0).sum())}/{len(reg)} | "
              f"subst.worse(>.10): {int((reg>.10).sum())}/{len(reg)}")
print(f"\ndomains: {sorted(set(r[0] for r in rows))}")
