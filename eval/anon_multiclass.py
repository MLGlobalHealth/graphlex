"""10-class ANONYMIZED structural classification — the convincing version.

Point: with 10 structurally-distinct graph families and ARBITRARY shuffled labels
(A..J, remapped per seed), the LLM cannot lean on "I know what scale-free is called."
The only way to beat chance (0.10) is to read the verbalized structure and learn each
of the 10 structure->label bindings from a few in-context examples. So accuracy >> 0.10
on anonymized labels is clean evidence of genuine in-context structural reasoning.

Families (all NetworkX, freshly sampled -> no instance leakage; sizes randomized so
node-count alone isn't the giveaway): ER, BA, Watts-Strogatz, random-regular,
random-geometric, SBM, Holme-Kim (powerlaw-cluster), 2D grid, caveman, random tree.

Arms: graphlex+LLM (few-shot ICL over the verbalized facts, ANON labels) and a non-LLM
anchor (logreg on the same feature_vector). Chance = 0.10.

Run: cd /home/scratch/fmsn-dev && source .venv/bin/activate && \
 PYTHONPATH=<graphlex> python eval/anon_multiclass.py
"""
import json, os, sys
import numpy as np
import networkx as nx
sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
from graphlex import facts, verbalize, feature_vector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = '/home/scratch/bench_out/anon_multiclass'
os.makedirs(OUT, exist_ok=True)
SEED = 7
K = 3          # shots / class
NQ_PER = 5     # queries / class  -> 50 queries
LETTERS = list("ABCDEFGHIJ")


def _clean(G):
    """Simple, undirected, self-loop-free, integer-labeled graph."""
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G)


def _int(G):
    return nx.convert_node_labels_to_integers(G)


def gen(fam, rng):
    """One freshly-sampled graph from family `fam`, size randomized."""
    n = int(rng.integers(45, 86))
    s = int(rng.integers(0, 1 << 30))
    if fam == "er":
        return nx.gnp_random_graph(n, 5.0 / (n - 1), seed=s)
    if fam == "ba":
        return nx.barabasi_albert_graph(n, 2, seed=s)
    if fam == "ws":
        return nx.watts_strogatz_graph(n, 6, 0.1, seed=s)
    if fam == "regular":
        n += (n * 4) % 2          # ensure n*d even
        return nx.random_regular_graph(4, n, seed=s)
    if fam == "rgg":
        return _int(nx.random_geometric_graph(n, np.sqrt(5.0 / (np.pi * n)), seed=s))
    if fam == "sbm":
        b = int(rng.integers(3, 5)); sizes = [n // b] * b
        P = np.full((b, b), 0.02); np.fill_diagonal(P, 0.35)
        return nx.stochastic_block_model(sizes, P, seed=s)
    if fam == "holme_kim":
        return nx.powerlaw_cluster_graph(n, 2, 0.45, seed=s)
    if fam == "grid":
        r = int(rng.integers(6, 10)); c = int(rng.integers(6, 10))
        return _int(nx.grid_2d_graph(r, c))
    if fam == "caveman":
        l = int(rng.integers(4, 8)); k = int(rng.integers(6, 11))
        return _int(nx.relaxed_caveman_graph(l, k, 0.1, seed=s))
    if fam == "tree":
        try:
            return nx.random_labeled_tree(n, seed=s)
        except AttributeError:
            return nx.random_tree(n, seed=s)
    raise ValueError(fam)


FAMILIES = ["er", "ba", "ws", "regular", "rgg", "sbm", "holme_kim", "grid", "caveman", "tree"]


def verbalize_graph(G):
    return verbalize(facts(G), focus="structure", groups="ABCDEFGHJ")


def main():
    rng = np.random.default_rng(SEED)
    # anonymization: family -> shuffled letter, this seed
    perm = list(rng.permutation(len(FAMILIES)))
    fam2letter = {FAMILIES[i]: LETTERS[perm[k]] for k, i in enumerate(perm)}
    # actually map each family to a distinct letter (random):
    letters_shuf = list(rng.permutation(LETTERS))
    fam2letter = {f: letters_shuf[i] for i, f in enumerate(FAMILIES)}

    def make(n_per):
        rows = []
        for f in FAMILIES:
            for _ in range(n_per):
                G = _clean(gen(f, rng))
                rows.append({"fam": f, "letter": fam2letter[f],
                             "verb": verbalize_graph(G), "x": feature_vector(facts(G))})
        return rows
    shots = make(K)
    queries = make(NQ_PER)
    rng.shuffle(queries)

    # non-LLM anchor: logreg on the same features
    Xs = np.array([r["x"] for r in shots]); ys = np.array([r["letter"] for r in shots])
    Xq = np.array([r["x"] for r in queries]); yq = np.array([r["letter"] for r in queries])
    sc = StandardScaler().fit(Xs)
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(Xs), ys)
    lr_acc = float((lr.predict(sc.transform(Xq)) == yq).mean())

    # build the anonymized few-shot prompt
    shot_txt = "\n\n".join(f"--- EXAMPLE {i} (Class {r['letter']}) ---\n{r['verb']}"
                           for i, r in enumerate(shots))
    q_txt = "\n\n".join(f"--- QUERY {i} ---\n{r['verb']}" for i, r in enumerate(queries))
    prompt = (
        f"You are classifying networks into {len(FAMILIES)} structural classes labeled "
        f"{', '.join(LETTERS)}. The class labels are ARBITRARY codes (a different random "
        f"assignment each run) — you must infer what structural pattern each letter denotes "
        f"from the labeled examples, then classify the queries. You are given only a "
        f"deterministic structural description of each network.\n\n"
        f"{K*len(FAMILIES)} labeled examples ({K} per class):\n\n{shot_txt}\n\n"
        f"Now classify these {len(queries)} networks. Output exactly one line per query as "
        f"'<id> <LETTER>'.\n\n{q_txt}")
    open(f"{OUT}/prompt.txt", "w").write(prompt)
    json.dump({"seed": SEED, "K": K, "n_classes": len(FAMILIES), "chance": 1 / len(FAMILIES),
               "fam2letter": fam2letter,
               "truth": {i: r["letter"] for i, r in enumerate(queries)},
               "truth_fam": {i: r["fam"] for i, r in enumerate(queries)},
               "logreg_acc": lr_acc},
              open(f"{OUT}/manifest.json", "w"), indent=1)

    print(f"10-class ANON structural classification (chance {1/len(FAMILIES):.2f})")
    print(f"  families -> letters: {fam2letter}")
    print(f"  shots {len(shots)} ({K}/class), queries {len(queries)} ({NQ_PER}/class)")
    print(f"  non-LLM anchor (logreg on same features): {lr_acc:.3f}")
    print(f"  prompt -> {OUT}/prompt.txt ({len(prompt)} chars)")
    print("\n  sanity — mean structural stats per family (first feature dims):")
    import collections
    F = facts(gen("tree", np.random.default_rng(1)))  # feature name order
    from graphlex import feature_names
    fn = feature_names()
    show = ["transitivity", "modularity", "n_cycles", "degree_gini", "max_kcore", "diameter"]
    idx = {k: fn.index(k) for k in show if k in fn}
    agg = collections.defaultdict(list)
    for r in shots + queries:
        agg[r["fam"]].append(r["x"])
    print("  fam        " + "  ".join(f"{k:>11}" for k in idx))
    for f in FAMILIES:
        m = np.mean(agg[f], 0)
        print(f"  {f:10s} " + "  ".join(f"{m[idx[k]]:>11.2f}" for k in idx))


if __name__ == "__main__":
    main()
