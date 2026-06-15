"""Node-level few-shot ICL track: graphlex+LLM vs trained logreg, on Planetoid.

Granularity-flexibility demo. The SAME facts()/verbalize() machinery that does
whole-graph classification (sweep.py, label_curve.py) is here pointed at a single
node's k-hop EGO-GRAPH. Each ICL "example" = one labeled node, rendered as its
verbalized ego-graph; the query = a target node's ego-graph, predict its class.
This lets graphlex+LLM span graph-level AND node-level in one table, where each
graph foundation model (PRODIGY, OFA) is locked to one granularity.

Mirrors label_curve.py exactly:
  * pure-ICL: writes seed*_k*.txt prompt files (LLM arm runs LATER via subagents /
    run_qwen.py; this script calls NO LLM). Same prompt skeleton: TASK header,
    "=== LABELED EXAMPLES ===" with [LABEL] blocks, "=== QUERIES ===" with
    "Query <id>:" blocks, "OUTPUT FORMAT: '<id> <CLASS>'".
  * class-balanced K-shot support: K labeled nodes PER CLASS (load-bearing rule).
  * >=3 seeds (load-bearing rule).
  * trained-logreg-on-node-features baseline at the SAME K labeled nodes/class,
    written into manifest.json alongside truth + chance (exactly like
    label_curve.py's logreg / sweep.py's baselines block).
  * answers scored later by score_node_icl.py via _common.parse_ans / bal_acc
    (BALANCED accuracy, load-bearing rule). See NODE_TRACK_PLAN.md.

EGO-GRAPH protocol: target node -> k_hop_subgraph (KHOP), capped at MAX_EGO nodes
(BFS-nearest kept, target always retained), -> networkx -> graphlex facts()/
verbalize(focus='structure'). The target node is identified in the prompt by its
relabeled id (always 0 here, since the target is placed first) AND an explicit
"TARGET NODE = #0" line, plus its own feature summary. Node bag-of-words features
are high-dim (1433 for Cora) with NO human-readable labels in the Planetoid loader,
so they are NOT verbalized as graphlex categorical node_attrs; instead each node's
top word-ids are summarized compactly (see node_feat_summary). The logreg baseline
uses the raw bag-of-words vectors (the standard Planetoid bar).

Run:  /home/scratch/fmsn-dev/.venv/bin/python eval/node_icl.py [DATASET]
      DATASET in {Cora, Citeseer, PubMed}; default Cora. Env SMOKE=1 -> small grid.
"""
import os, sys, json
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph, to_networkx
import torch

sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
from graphlex import facts, verbalize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from node_text import load_readable, node_text_summary   # arm (b) readable text

ROOT = '/home/scratch/planetoid'
OUT_BASE = '/home/scratch/bench_out/node_icl'
SEEDS = [11, 22, 33]            # >=3 seeds (load-bearing)
K_SHOTS = [1, 3, 5]            # labeled nodes PER CLASS (class-balanced, load-bearing)
KHOP = 2                        # ego-graph radius (1 or 2; see NODE_TRACK_PLAN.md)
MAX_EGO = 25                    # cap ego-graph node count (BFS-nearest kept)
NQ = 30                         # query nodes (class-balanced across the dataset's classes)
TOPW = 12                       # top word-ids shown per node (opaque arm)
# Two node-feature representations, run + compared (see NODE_TRACK_PLAN.md §2):
#   'opaque'   -> [w26,w61,...] anonymized word-ids (Planetoid native)
#   'readable' -> real paper title+abstract (TAG/Graph-LLM release, node-aligned)
REPS = ['opaque', 'readable']

if os.environ.get('SMOKE'):
    SEEDS, K_SHOTS, NQ = [11, 22, 33], [1, 3, 5], 30  # smoke == default small grid


# --- ego-graph extraction -----------------------------------------------------
def ego_nx(data, target, khop, max_ego):
    """k-hop ego-graph of `target` as a networkx graph, capped to <=max_ego nodes
    (BFS-nearest to target kept; target always retained and relabeled to id 0).
    Returns (G, global_node_ids) where G is relabeled 0..n-1 with G's node 0 = target."""
    sub_nodes, _, mapping, _ = k_hop_subgraph(
        int(target), khop, data.edge_index, relabel_nodes=False,
        num_nodes=data.num_nodes)
    glob = sub_nodes.tolist()
    if target not in glob:                     # isolated node: just itself
        glob = [int(target)]
    # BFS-distance ordering from target for the cap (keep nearest)
    if len(glob) > max_ego:
        gset = set(glob)
        # build adjacency within the subgraph for a quick BFS
        full = to_networkx(data, to_undirected=True)
        order, seen = [int(target)], {int(target)}
        frontier = [int(target)]
        while frontier and len(order) < max_ego:
            nxt = []
            for u in frontier:
                for w in full.neighbors(u):
                    if w in gset and w not in seen:
                        seen.add(w); order.append(w); nxt.append(w)
                        if len(order) >= max_ego:
                            break
                if len(order) >= max_ego:
                    break
            frontier = nxt
        glob = order
    glob = [int(target)] + [g for g in glob if g != int(target)]   # target first
    idx = {g: i for i, g in enumerate(glob)}
    G = nx.Graph()
    G.add_nodes_from(range(len(glob)))
    ei = data.edge_index.numpy()
    gset = set(glob)
    for a, b in zip(ei[0], ei[1]):
        a, b = int(a), int(b)
        if a in gset and b in gset and a != b:
            G.add_edge(idx[a], idx[b])
    return G, glob


def node_feat_summary(data, glob, topw):
    """Compact per-node feature summary line for the ego-graph (high-dim BoW has no
    human-readable labels in the Planetoid loader -> show top active feature ids)."""
    x = data.x.numpy()
    lines = []
    for local, g in enumerate(glob):
        nz = np.nonzero(x[g])[0]
        vals = x[g][nz]
        order = nz[np.argsort(-vals)][:topw]
        tag = "TARGET" if local == 0 else f"#{local}"
        feats = ",".join(f"w{int(w)}" for w in order) if len(order) else "(none)"
        lines.append(f"  {tag}: features [{feats}]")
    return "\n".join(lines)


def verbalize_ego(data, target, khop, max_ego, topw, rep='opaque', texts=None):
    """Verbalize a node's ego-graph. rep='opaque' -> top word-id summary (Planetoid
    native); rep='readable' -> real paper title+abstract per node (TAG release)."""
    G, glob = ego_nx(data, target, khop, max_ego)
    struct = verbalize(facts(G), focus='structure')
    if rep == 'readable':
        feats = node_text_summary(texts, glob)
        flabel = "Node text (title/abstract)"
    else:
        feats = node_feat_summary(data, glob, topw)
        flabel = "Node features (top word-ids)"
    return (f"TARGET NODE = #0 (its {khop}-hop neighborhood below; node #0 is the "
            f"node to classify).\nEgo-graph: {struct}\n{flabel}:\n{feats}")


# --- class-balanced K-shot node splits ----------------------------------------
def make_splits(y, classes, seeds, k_max, nq, rng_pool):
    """For each seed: K labeled nodes/class (K up to k_max, nested so K=1 subset of
    K=3 subset of K=5) + nq class-balanced query nodes, disjoint from all shots."""
    out = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        per = {c: list(rng.permutation([i for i in rng_pool if y[i] == c])) for c in classes}
        shots = {c: per[c][:k_max] for c in classes}            # first k_max per class
        used = set(i for c in classes for i in shots[c])
        # class-balanced queries from the remainder
        rem = {c: [i for i in per[c][k_max:]] for c in classes}
        nper = max(1, nq // len(classes))
        qids = []
        for c in classes:
            qids += rem[c][:nper]
        qids = qids[:nq]
        rng.shuffle(qids)
        out[seed] = {"shots": shots, "queries": qids}
    return out


# --- trained-logreg baseline on raw node features -----------------------------
def logreg_at(data, y, shots, queries, k):
    """logreg on raw node bag-of-words features, trained on K labeled nodes/class,
    evaluated on the query nodes. Returns balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score
    x = data.x.numpy()
    tr_ids = [i for c, lst in shots.items() for i in lst[:k]]
    ytr = y[tr_ids]
    if len(set(ytr.tolist())) < 2:
        return None
    Xtr = x[tr_ids]
    Xq = x[queries]; yq = y[queries]
    sc = StandardScaler(with_mean=False).fit(Xtr)   # sparse-friendly
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xq))
    return float(balanced_accuracy_score(yq, pred))


# --- matched few-shot trained-GNN baseline (GCN) ------------------------------
# Standard 2-layer semi-supervised node GCN, trained on the SAME K labeled
# nodes/class as the LLM/logreg arms, evaluated on the SAME query nodes, BALANCED
# accuracy. Uses the GCNConv from gnn_baseline.py's family (kept local + tiny so
# this runs on CPU). Expected weak at K=1..5 -- reported honestly (NODE_TRACK_PLAN).
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

GCN_HIDDEN = 64
GCN_LAYERS = 2
GCN_DROPOUT = 0.5
GCN_LR = 1e-2
GCN_WD = 5e-4
GCN_EPOCHS = 200
GCN_PATIENCE = 30


class NodeGCN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, n_layers=GCN_LAYERS, dropout=GCN_DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else hidden,
                                      hidden if i < n_layers - 1 else n_classes))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def gcn_at(data, y, shots, queries, k, classes, seed):
    """Few-shot node GCN: train on K labeled nodes/class over the FULL graph,
    eval balanced acc on the query nodes. Same labeled set as logreg/LLM arms."""
    from sklearn.metrics import balanced_accuracy_score
    tr_ids = [i for c, lst in shots.items() for i in lst[:k]]
    ytr = y[tr_ids]
    if len(set(ytr.tolist())) < 2:
        return None
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = data.x.float().to(dev)
    ei = data.edge_index.to(dev)
    yt = torch.as_tensor(y, dtype=torch.long).to(dev)
    cls_sorted = sorted(classes)
    remap = {c: i for i, c in enumerate(cls_sorted)}
    yr = torch.as_tensor([remap[int(v)] for v in y], dtype=torch.long).to(dev)
    tr = torch.as_tensor(tr_ids, dtype=torch.long).to(dev)
    # class-weighted loss to counter any shot imbalance (here balanced by design)
    bc = np.bincount([remap[int(v)] for v in ytr], minlength=len(cls_sorted)).astype(float)
    cw = torch.tensor(np.where(bc > 0, bc.sum() / (len(bc) * bc), 1.0),
                      dtype=torch.float, device=dev)
    model = NodeGCN(x.size(1), GCN_HIDDEN, len(cls_sorted)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=GCN_LR, weight_decay=GCN_WD)
    crit = nn.CrossEntropyLoss(weight=cw)
    best_tr, best_state, since = 1e9, None, 0
    for ep in range(GCN_EPOCHS):
        model.train(); opt.zero_grad()
        out = model(x, ei)
        loss = crit(out[tr], yr[tr])
        loss.backward(); opt.step()
        # early-stop on (tiny) train loss plateau -- no held-out set at K=1
        lv = float(loss.detach())
        if lv < best_tr - 1e-4:
            best_tr, since = lv, 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            since += 1
            if since >= GCN_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x, ei).argmax(1).cpu().numpy()
    inv = {i: c for c, i in remap.items()}
    pq = np.array([inv[int(pred[q])] for q in queries])
    return float(balanced_accuracy_score(y[queries], pq))


def build_prompt(data, y, classes, clsname, token, shots, queries, k, khop, max_ego,
                 topw, rep='opaque', texts=None):
    """Build one ICL prompt. clsname[c] = human-readable topic shown in the TASK;
    token[c] = the SHORT, UNAMBIGUOUS label emitted/parsed in answers (CLASS0..)."""
    what = ("each node's full paper title and abstract" if rep == 'readable'
            else "each node's most active (anonymized) word features")
    topics = ", ".join(f"{token[c]} = {clsname[c]}" for c in classes)
    TASK = (f"Each item is a NODE in a citation network, shown as its {khop}-hop "
            f"ego-graph (the node plus its nearby neighbors) with {what}. Classify "
            f"the TARGET node (#0) into one of {len(classes)} topics:\n  {topics}\n"
            f"Learn the pattern from the labeled examples, then classify each query.\n"
            f"OUTPUT FORMAT: one line per query, exactly '<id> <CLASS>' where <CLASS> "
            f"is one of {', '.join(token[c] for c in classes)}. No other text.")
    L = [TASK, "", "=== LABELED EXAMPLES ==="]
    for c in classes:
        for v in shots[c][:k]:
            L.append(f"[{token[c]}]\n{verbalize_ego(data, v, khop, max_ego, topw, rep, texts)}\n")
    L.append("=== QUERIES (classify each) ===")
    truth = []
    for qi, v in enumerate(queries):
        L.append(f"Query {qi}:\n{verbalize_ego(data, v, khop, max_ego, topw, rep, texts)}\n")
        truth.append([qi, token[int(y[v])]])
    return "\n".join(L), truth


def run(dataset):
    out = f"{OUT_BASE}/{dataset}"
    ds = Planetoid(ROOT, name=dataset)
    data = ds[0]
    y = data.y.numpy()
    classes = sorted(set(y.tolist()))
    token = {c: f"CLASS{c}" for c in classes}     # answer token (parsed by scorer)
    pool = list(range(data.num_nodes))
    splits = make_splits(y, classes, SEEDS, max(K_SHOTS), NQ, pool)

    # readable text + real class names (arm b); class names also enrich every prompt
    texts, real_names, align = load_readable(dataset)
    clsname = {c: real_names.get(c, f"CLASS{c}") for c in classes}

    # logreg + GCN are representation-INDEPENDENT (both train on the raw Planetoid
    # feature matrix), so compute them ONCE per (seed,k) and store at manifest top.
    lrc = {str(k): [] for k in K_SHOTS}
    gcc = {str(k): [] for k in K_SHOTS}
    for seed in SEEDS:
        sh = splits[seed]["shots"]; q = splits[seed]["queries"]
        for k in K_SHOTS:
            v = logreg_at(data, y, sh, q, k)
            if v is not None:
                lrc[str(k)].append(v)
            g = gcn_at(data, y, sh, q, k, classes, seed)
            if g is not None:
                gcc[str(k)].append(g)

    # prompt files PER REPRESENTATION
    rep_files = {}
    for rep in REPS:
        os.makedirs(f"{OUT_BASE}/{dataset}/{rep}/ans/opus", exist_ok=True)
        files = {}
        for seed in SEEDS:
            sh = splits[seed]["shots"]; q = splits[seed]["queries"]
            support_ids = {token[c]: [int(i) for i in sh[c][:max(K_SHOTS)]] for c in classes}
            for k in K_SHOTS:
                prompt, truth = build_prompt(data, y, classes, clsname, token, sh, q, k,
                                             KHOP, MAX_EGO, TOPW, rep=rep, texts=texts)
                fn = f"{dataset}/{rep}/seed{seed}_k{k}.txt"
                open(f"{OUT_BASE}/{fn}", 'w').write(prompt)
                files[fn] = {"dataset": dataset, "rep": rep, "seed": seed, "k": k,
                             "truth": truth, "khop": KHOP, "max_ego": MAX_EGO,
                             "query_ids": [int(i) for i in q],
                             "support_ids": {c: ids[:k] for c, ids in support_ids.items()}}
        rep_files[rep] = files

    man = {"dataset": dataset, "seeds": SEEDS, "k_shots": K_SHOTS, "khop": KHOP,
           "max_ego": MAX_EGO, "nq": NQ, "n_classes": len(classes),
           "chance": 1.0 / len(classes), "representations": REPS,
           "class_names": {token[c]: clsname[c] for c in classes},
           "text_alignment": align,
           "files": {fn: m for fl in rep_files.values() for fn, m in fl.items()},
           "logreg": {k: v for k, v in lrc.items()},
           "gcn": {k: v for k, v in gcc.items()},
           "metric": "balanced_accuracy"}
    json.dump(man, open(f"{out}/manifest.json", 'w'), indent=0)
    print(f"[{dataset}] {len(classes)} classes, chance {1.0/len(classes):.3f}, "
          f"khop={KHOP}, max_ego={MAX_EGO}, nq={NQ}; text={align}")
    print(f"  logreg (bal acc):",
          {k: f"{np.mean(v):.3f}+/-{np.std(v):.3f}" for k, v in lrc.items() if v})
    print(f"  GCN    (bal acc):",
          {k: f"{np.mean(v):.3f}+/-{np.std(v):.3f}" for k, v in gcc.items() if v})
    n = sum(len(fl) for fl in rep_files.values())
    print(f"  wrote {n} prompt files ({len(REPS)} reps) + manifest -> {out}")
    return man


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'Cora'
    os.makedirs(OUT_BASE, exist_ok=True)
    run(dataset)
    print("run subagents (or run_qwen.py) on each seed*_k*.txt; score with score_node_icl.py")
