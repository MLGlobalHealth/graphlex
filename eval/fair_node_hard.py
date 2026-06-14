"""Fair relational node prediction WITHOUT the tautology.

The pilot's fair_node_pred.py handed the model a pre-aggregated tally
("known colleagues by department: CS:3, Math:1, ...") -- i.e. the exact statistic
neighbor-majority takes the argmax of. So "verbalize+Claude = neighbor-majority"
was a lookup, not reasoning. Here we emit TWO arms per seed so the confound is
measurable:

  tally  : the original pre-aggregated dept counts (handed the answer)   [confounded]
  nolist : NO tally. Unaggregated neighbor-department list with some neighbors
           UNLABELLED ('?'); the model must aggregate itself and tolerate missing
           labels (a mild collective-classification step).               [de-confounded]

Lower labelled fraction (50%) so neighbor-majority is imperfect -> room for the LLM
to beat OR miss it. Baselines (neighbor-majority over known, logreg) computed here.

Run:  /home/scratch/fmsn-dev/.venv/bin/python eval/fair_node_hard.py
"""
import os, json
import numpy as np
import networkx as nx
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = '/home/scratch/bench_out/fairnode_v2'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/ans/opus", exist_ok=True)
SEEDS = [11, 22, 33]
DEPTS = ['CS', 'Math', 'Bio', 'Eng']
SIZES = [30, 30, 30, 30]
P = np.full((4, 4), 0.02)
np.fill_diagonal(P, 0.12)
KNOWN_FRAC = 0.50


def build(seed):
    rng = np.random.RandomState(seed)
    G = nx.stochastic_block_model(SIZES, P, seed=seed)
    lab, b = {}, 0
    for k, s in enumerate(SIZES):
        for _ in range(s):
            lab[b] = k; b += 1
    nodes = list(G.nodes()); rng.shuffle(nodes)
    nk = int(KNOWN_FRAC * len(nodes))
    known = set(nodes[:nk]); held = nodes[nk:]
    clust = nx.clustering(G)

    def kcounts(v):  # counts over KNOWN neighbors only
        c = Counter(lab[u] for u in G.neighbors(v) if u in known)
        return [c.get(k, 0) for k in range(4)]

    glob = Counter(lab[v] for v in known).most_common(1)[0][0]

    def majority(v):
        c = kcounts(v)
        return int(np.argmax(c)) if sum(c) else glob
    acc_major = float(np.mean([majority(v) == lab[v] for v in held]))

    def feats(v):
        return [G.degree(v), clust[v]] + kcounts(v)
    Xtr = np.array([feats(v) for v in known]); ytr = np.array([lab[v] for v in known])
    Xte = np.array([feats(v) for v in held]); yte = np.array([lab[v] for v in held])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    acc_lr = float((clf.predict(sc.transform(Xte)) == yte).mean())

    # ---- prompt descriptions ----
    def desc_tally(v):
        c = kcounts(v)
        cs = ", ".join(f"{DEPTS[k]}:{c[k]}" for k in range(4))
        return (f"- {G.degree(v)} connections, clustering {clust[v]:.2f}; "
                f"known colleagues by department: {cs}")

    def desc_nolist(v):
        # unaggregated list; unknown neighbors shown as '?'  (model must aggregate)
        items = []
        for u in G.neighbors(v):
            items.append(DEPTS[lab[u]] if u in known else "?")
        rng.shuffle(items)
        return (f"- {G.degree(v)} connections, clustering {clust[v]:.2f}; "
                f"colleagues' departments (some unknown): [{', '.join(items)}]")

    shot_nodes = []
    by = {k: [v for v in known if lab[v] == k] for k in range(4)}
    for k in range(4):
        shot_nodes += list(rng.permutation(by[k]))[:6]
    rng.shuffle(shot_nodes)

    TASK = ("Predict each person's department (one of: CS, Math, Bio, Eng) in a "
            "university collaboration network where colleagues tend to share a "
            "department. Use the labeled examples to learn the pattern, then "
            "classify each query.\nOUTPUT FORMAT: one line per query, exactly "
            "'<id> <DEPT>'. No other text.")

    out = {}
    for arm, desc in [('tally', desc_tally), ('nolist', desc_nolist)]:
        L = [TASK, "", "=== LABELED EXAMPLES ==="]
        for v in shot_nodes:
            L.append(f"[{DEPTS[lab[v]]}]\n{desc(v)}\n")
        L.append("=== QUERIES (classify each) ===")
        truth = []
        for q, v in enumerate(held):
            L.append(f"Query {q}:\n{desc(v)}\n")
            truth.append([q, DEPTS[lab[v]]])
        fn = f"seed{seed}_{arm}.txt"
        open(f"{OUT}/{fn}", 'w').write("\n".join(L))
        out[fn] = {"arm": arm, "seed": seed, "truth": truth}
    return out, acc_major, acc_lr, len(held)


if __name__ == "__main__":
    man = {"seeds": SEEDS, "chance": 0.25, "known_frac": KNOWN_FRAC,
           "files": {}, "neighbor_majority": {}, "logreg": {}, "nq": None}
    for s in SEEDS:
        files, am, lr, nq = build(s)
        man["files"].update(files)
        man["neighbor_majority"][str(s)] = am
        man["logreg"][str(s)] = lr
        man["nq"] = nq
        print(f"seed {s}: nq={nq}, neighbor-majority={am:.3f}, logreg={lr:.3f}")
    json.dump(man, open(f"{OUT}/manifest.json", 'w'), indent=0)
    nm = list(man["neighbor_majority"].values()); lrs = list(man["logreg"].values())
    print(f"\nneighbor-majority: {np.mean(nm):.3f} +/- {np.std(nm):.3f}")
    print(f"logreg          : {np.mean(lrs):.3f} +/- {np.std(lrs):.3f}  (chance 0.25)")
    print(f"manifest -> {OUT}/manifest.json")
