"""Edge / link-prediction few-shot ICL track: graphlex+LLM vs classical-heuristic
logreg vs a trained GNN link predictor, on Planetoid.

THIRD task granularity. After whole-graph classification (sweep.py / label_curve.py)
and node classification (node_icl.py), this points the SAME facts()/verbalize()
machinery at a NODE PAIR (u,v) and asks: is there an edge between them (LINK) or not
(NOLINK)? This completes the granularity-flexibility story: graphlex+LLM spans
graph-, node-, AND edge-level in one results table, where each specialist graph
foundation model is locked to a single granularity (PRODIGY/OFA node, GraphPFN node,
ULTRA link). See EDGE_TRACK_PLAN.md.

Closely parallels node_icl.py — read that first. The differences are:
  * a "query" is a NODE PAIR (u,v); label = edge present (LINK / CLASS1) or absent
    (NOLINK / CLASS0). BALANCED query set: NQ/2 positive edges + NQ/2 negative
    non-edges (standard link-prediction protocol).
  * few-shot support = K positive + K negative example pairs (class-balanced), each
    verbalized.
  * VERBALIZATION (the graphlex angle): for a pair (u,v) we verbalize their JOINT
    neighborhood — the union of u's and v's k-hop ego-graphs, with BOTH endpoints
    clearly marked (TARGET PAIR = #0 and #1) — via graphlex facts()/verbalize(),
    PLUS computed classical link features rendered as a readable line: common-neighbor
    count, Jaccard, Adamic-Adar, resource-allocation, preferential-attachment,
    shortest-path length, same-community. The link features are computed with
    networkx and APPENDED as a line (parallel to how node_icl appends the per-node
    readable line) — graphlex core is NOT modified.
  * BOTH 'opaque' (word-id node summaries) and 'readable' (title/abstract) reps.

IMPORTANT — train/eval edge split. To avoid label leakage, the positive QUERY edges
and the positive SUPPORT edges are REMOVED from the graph that is used to (a) compute
the link-feature heuristics, (b) build the verbalized neighborhoods, and (c) train
the GNN link predictor. So no method gets to see the edge it must predict. This is
the standard link-prediction observed-graph protocol. The negative pairs are sampled
from the complement (true non-edges of the FULL graph).

Metrics: BALANCED accuracy (consistency with the rest of the suite) AND AUC
(standard for link prediction), mean +/- std over seeds, into manifest.json.

Run:  /home/scratch/fmsn-dev/.venv/bin/python eval/edge_icl.py [DATASET]
      DATASET in {Cora, Citeseer, PubMed}; default Cora. Calls NO LLM.
"""
import os, sys, json
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import torch

sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
from graphlex import facts, verbalize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from node_text import load_readable, node_text_summary   # readable rep text

ROOT = '/home/scratch/planetoid'
OUT_BASE = '/home/scratch/bench_out/edge_icl'
SEEDS = [11, 22, 33]            # >=3 seeds (load-bearing)
K_SHOTS = [1, 3, 5]            # pos/neg example pairs PER CLASS (class-balanced, load-bearing)
KHOP = 1                        # ego-graph radius per endpoint for the JOINT neighborhood
MAX_JOINT = 30                  # cap joint-neighborhood node count (BFS-nearest kept)
NQ = 40                         # query pairs (NQ/2 positive edges + NQ/2 negatives)
TOPW = 10                       # top word-ids shown per node (opaque rep)
REPS = ['opaque', 'readable']
# link tokens: reuse CLASS0/CLASS1 so _common.parse_ans + the existing drivers work.
# CLASS1 = LINK (edge present), CLASS0 = NOLINK (edge absent).
TOK_POS, TOK_NEG = 'CLASS1', 'CLASS0'

if os.environ.get('SMOKE'):
    SEEDS, K_SHOTS, NQ = [11, 22, 33], [1, 3], 12


# --- edge / non-edge sampling -------------------------------------------------
def sample_pairs(G, n_pos, n_neg, rng):
    """Sample n_pos existing edges (as positive pairs) + n_neg true non-edges
    (negative pairs) from undirected graph G. Returns (pos_list, neg_list) of
    (u,v) tuples with u<v. Positives drawn from G.edges; negatives rejection-sampled
    from the complement."""
    edges = [(int(min(u, v)), int(max(u, v))) for u, v in G.edges() if u != v]
    edges = list(set(edges))
    rng.shuffle(edges)
    pos = edges[:n_pos]
    eset = set(edges)
    nodes = list(G.nodes())
    neg, seen = [], set()
    # only sample negatives between nodes that have at least one edge, so the pair
    # actually sits in the graph (isolated nodes give empty neighborhoods).
    cand = [n for n in nodes if G.degree(n) > 0]
    tries = 0
    while len(neg) < n_neg and tries < n_neg * 2000:
        tries += 1
        u = int(cand[rng.randint(len(cand))])
        v = int(cand[rng.randint(len(cand))])
        if u == v:
            continue
        a, b = min(u, v), max(u, v)
        if (a, b) in eset or (a, b) in seen:
            continue
        seen.add((a, b)); neg.append((a, b))
    return pos, neg


# --- joint k-hop neighborhood extraction --------------------------------------
def joint_nx(G, u, v, khop, max_joint):
    """Union of u's and v's k-hop ego-graphs as a networkx graph, capped to
    <=max_joint nodes (BFS-nearest to {u,v} kept; u,v always retained and relabeled
    to ids 0,1). Edges are taken from G EXCEPT the (u,v) edge itself is never added
    (caller passes an observed graph with query/support positives already removed,
    but we strip (u,v) defensively too). Returns (H, glob) with glob[0]=u, glob[1]=v."""
    def khop_nodes(src):
        seen = {src}; frontier = {src}
        for _ in range(khop):
            nxt = set()
            for x in frontier:
                nxt |= set(G.neighbors(x))
            nxt -= seen
            seen |= nxt; frontier = nxt
        return seen
    nodes = khop_nodes(u) | khop_nodes(v)
    nodes.discard(u); nodes.discard(v)
    rest = list(nodes)
    # BFS-distance cap: keep nodes nearest to either endpoint
    if len(rest) > max_joint - 2:
        dist = {}
        for src in (u, v):
            d = {src: 0}; frontier = [src]
            for hop in range(1, khop + 1):
                nf = []
                for x in frontier:
                    for w in G.neighbors(x):
                        if w not in d:
                            d[w] = hop; nf.append(w)
                frontier = nf
            for w, dd in d.items():
                if w not in (u, v):
                    dist[w] = min(dist.get(w, 99), dd)
        rest = [w for w, _ in sorted(dist.items(), key=lambda kv: kv[1])][:max_joint - 2]
    glob = [u, v] + rest
    idx = {g: i for i, g in enumerate(glob)}
    gset = set(glob)
    H = nx.Graph()
    H.add_nodes_from(range(len(glob)))
    for a in glob:
        for b in G.neighbors(a):
            if b in gset and a != b:
                ai, bi = idx[a], idx[b]
                if {ai, bi} == {0, 1}:        # never reveal the query edge
                    continue
                H.add_edge(ai, bi)
    return H, glob


# --- classical link features (networkx) ---------------------------------------
LINK_FEATS = ['common_neighbors', 'jaccard', 'adamic_adar', 'resource_allocation',
              'preferential_attachment', 'shortest_path', 'same_community']


def _community_of(G):
    """Greedy-modularity community id per node (computed once per observed graph)."""
    try:
        comms = nx.community.greedy_modularity_communities(G)
    except Exception:
        return {}
    cid = {}
    for c, nodes in enumerate(comms):
        for n in nodes:
            cid[n] = c
    return cid


def link_features(G, u, v, comm):
    """Classical link-prediction feature vector for pair (u,v) on observed graph G
    (G must NOT contain the (u,v) edge). Returns a dict keyed by LINK_FEATS."""
    nu, nv = set(G.neighbors(u)) if u in G else set(), set(G.neighbors(v)) if v in G else set()
    nu.discard(v); nv.discard(u)
    cn = len(nu & nv)
    try:
        jac = next(iter(nx.jaccard_coefficient(G, [(u, v)])))[2]
    except Exception:
        jac = 0.0
    try:
        aa = next(iter(nx.adamic_adar_index(G, [(u, v)])))[2]
    except Exception:
        aa = 0.0
    try:
        ra = next(iter(nx.resource_allocation_index(G, [(u, v)])))[2]
    except Exception:
        ra = 0.0
    try:
        pa = next(iter(nx.preferential_attachment(G, [(u, v)])))[2]
    except Exception:
        pa = float(len(nu) * len(nv))
    try:
        sp = nx.shortest_path_length(G, u, v)
    except Exception:
        sp = -1                      # disconnected -> sentinel (no path)
    same = 1 if (comm.get(u) is not None and comm.get(u) == comm.get(v)) else 0
    return {'common_neighbors': float(cn), 'jaccard': float(jac),
            'adamic_adar': float(aa), 'resource_allocation': float(ra),
            'preferential_attachment': float(pa), 'shortest_path': float(sp),
            'same_community': float(same)}


def feat_vec(d):
    return [d[k] for k in LINK_FEATS]


def link_feat_line(d):
    """Readable line of the computed link features (appended to the prompt, parallel
    to node_icl's per-node readable line). graphlex core untouched."""
    sp = int(d['shortest_path'])
    sp_s = "no path (disconnected in observed graph)" if sp < 0 else f"{sp}"
    return ("Link features (computed on the observed graph, excluding this pair): "
            f"common-neighbors={int(d['common_neighbors'])}, "
            f"Jaccard={d['jaccard']:.3f}, Adamic-Adar={d['adamic_adar']:.3f}, "
            f"resource-allocation={d['resource_allocation']:.3f}, "
            f"preferential-attachment={int(d['preferential_attachment'])}, "
            f"shortest-path-length={sp_s}, "
            f"same-community={'yes' if d['same_community'] else 'no'}")


# --- per-node feature summaries (opaque / readable) ---------------------------
def node_feat_summary(data, glob, topw):
    """Compact per-node word-id summary (opaque rep). #0 and #1 are the target pair."""
    x = data.x.numpy()
    lines = []
    for local, g in enumerate(glob):
        nz = np.nonzero(x[g])[0]
        vals = x[g][nz]
        order = nz[np.argsort(-vals)][:topw]
        tag = "TARGET-A (#0)" if local == 0 else ("TARGET-B (#1)" if local == 1 else f"#{local}")
        feats = ",".join(f"w{int(w)}" for w in order) if len(order) else "(none)"
        lines.append(f"  {tag}: features [{feats}]")
    return "\n".join(lines)


def node_text_summary_pair(texts, glob, maxchars=260):
    """Readable per-node title/abstract line; #0/#1 are the target pair."""
    lines = []
    for local, g in enumerate(glob):
        t = (texts[g] or "").strip().replace('\n', ' ')
        if len(t) > maxchars:
            t = t[:maxchars].rsplit(' ', 1)[0] + ' ...'
        tag = "TARGET-A (#0)" if local == 0 else ("TARGET-B (#1)" if local == 1 else f"#{local}")
        lines.append(f"  {tag}: {t}")
    return "\n".join(lines)


def verbalize_pair(G_obs, data, u, v, comm, khop, max_joint, topw, rep='opaque', texts=None):
    """Verbalize a node pair's JOINT neighborhood + computed link features. G_obs is
    the observed graph (query/support positives removed); the (u,v) edge is never
    shown. #0 = u, #1 = v."""
    H, glob = joint_nx(G_obs, u, v, khop, max_joint)
    struct = verbalize(facts(H), focus='structure')
    if rep == 'readable':
        feats = node_text_summary_pair(texts, glob)
        flabel = "Node text (title/abstract)"
    else:
        feats = node_feat_summary(data, glob, topw)
        flabel = "Node features (top word-ids)"
    lf = link_features(G_obs, u, v, comm)
    return (f"TARGET PAIR = #0 and #1 (predict whether an edge connects them).\n"
            f"Joint {khop}-hop neighborhood of the pair: {struct}\n"
            f"{flabel}:\n{feats}\n{link_feat_line(lf)}"), lf


# --- balanced pos/neg pair splits ---------------------------------------------
def make_splits(G, seeds, k_max, nq):
    """Per seed: k_max positive + k_max negative SUPPORT pairs, plus nq/2 positive +
    nq/2 negative QUERY pairs, all disjoint. Supports nested in K (K=1 subset of K=3
    subset of K=5). Returns {seed: {pos_shots, neg_shots, q_pos, q_neg}}."""
    out = {}
    nqh = nq // 2
    for seed in seeds:
        rng = np.random.RandomState(seed)
        pos, neg = sample_pairs(G, k_max + nqh, k_max + nqh, rng)
        pos_shots, neg_shots = pos[:k_max], neg[:k_max]
        q_pos, q_neg = pos[k_max:k_max + nqh], neg[k_max:k_max + nqh]
        out[seed] = {"pos_shots": pos_shots, "neg_shots": neg_shots,
                     "q_pos": q_pos, "q_neg": q_neg}
    return out


def observed_graph(G_full, removed_pos):
    """Copy of the full graph with the given positive pairs (support + query
    positives) removed, so no method sees the edge it must predict."""
    G = G_full.copy()
    for (u, v) in removed_pos:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    return G


# --- baseline 1: classical-heuristic logreg -----------------------------------
def heuristic_logreg(G_obs, comm, pos_shots, neg_shots, q_pos, q_neg, k):
    """Train logreg on the link-feature vectors of K pos + K neg support pairs,
    evaluate on the balanced query pairs. Returns (balanced_acc, auc). All features
    computed on the observed graph (positives removed)."""
    tr_pairs = pos_shots[:k] + neg_shots[:k]
    ytr = [1] * k + [0] * k
    if len(set(ytr)) < 2:
        return None, None
    Xtr = np.array([feat_vec(link_features(G_obs, u, v, comm)) for (u, v) in tr_pairs])
    q_pairs = q_pos + q_neg
    yq = np.array([1] * len(q_pos) + [0] * len(q_neg))
    Xq = np.array([feat_vec(link_features(G_obs, u, v, comm)) for (u, v) in q_pairs])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xq))
    prob = clf.predict_proba(sc.transform(Xq))[:, 1]
    ba = float(balanced_accuracy_score(yq, pred))
    try:
        auc = float(roc_auc_score(yq, prob))
    except Exception:
        auc = None
    return ba, auc


# --- baseline 2: trained GNN link predictor (GCN encoder + dot-product) --------
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

GCN_HIDDEN = 64
GCN_OUT = 32
GCN_DROPOUT = 0.5
GCN_LR = 1e-2
GCN_WD = 5e-4
GCN_EPOCHS = 200
GCN_PATIENCE = 30


class GCNEncoder(nn.Module):
    """2-layer GCN encoder; the decoder is a dot product of endpoint embeddings.
    Standard GAE-style link predictor."""
    def __init__(self, in_dim, hidden=GCN_HIDDEN, out=GCN_OUT, dropout=GCN_DROPOUT):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, out)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.c2(x, edge_index)


def _edge_index_from_obs(G_obs, num_nodes, dev):
    """Undirected edge_index of the observed graph (both directions)."""
    src, dst = [], []
    for u, v in G_obs.edges():
        src += [u, v]; dst += [v, u]
    if not src:
        return torch.zeros((2, 0), dtype=torch.long, device=dev)
    return torch.tensor([src, dst], dtype=torch.long, device=dev)


def gnn_link(data, G_obs, pos_shots, neg_shots, q_pos, q_neg, k, seed):
    """Standard GAE-style link predictor: a GCN encoder + dot-product decoder trained
    on the OBSERVED graph's edges (all of them) as positive supervision with an equal
    number of randomly-sampled negative non-edges per epoch, evaluated on the balanced
    query pairs. Returns (balanced_acc, auc).

    This is the SPECIALIST link-prediction bar (the standard self-supervised GAE
    protocol), parallel to how the node-track GCN exploits the full graph: it learns
    from ALL observed edges, not just the K few-shot support pairs (a dot-product
    decoder cannot be fit from 2..10 pairs). It is therefore K-INDEPENDENT; we still
    report it per K row for table alignment (identical value across K, query set is
    the same). Message-passing + supervision graph = observed graph (query + support
    positives removed -> no leakage). Evaluated on the identical query pairs."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = data.x.float().to(dev)
    ei = _edge_index_from_obs(G_obs, data.num_nodes, dev)
    # positive supervision = all observed (undirected) edges, deduped to u<v
    pos_e = list({(int(min(u, v)), int(max(u, v))) for u, v in G_obs.edges() if u != v})
    if len(pos_e) < 2:
        return None, None
    pos_t = torch.tensor([[u for u, v in pos_e], [v for u, v in pos_e]],
                         dtype=torch.long, device=dev)
    eset = set(pos_e)
    n_nodes = data.num_nodes
    rng = np.random.RandomState(seed)
    model = GCNEncoder(x.size(1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=GCN_LR, weight_decay=GCN_WD)
    best, best_state, since = 1e9, None, 0
    for ep in range(GCN_EPOCHS):
        model.train(); opt.zero_grad()
        z = model(x, ei)
        # fresh random negatives each epoch (standard GAE negative sampling)
        nu = rng.randint(0, n_nodes, size=len(pos_e))
        nv = rng.randint(0, n_nodes, size=len(pos_e))
        neg_t = torch.tensor([nu.tolist(), nv.tolist()], dtype=torch.long, device=dev)
        pl = (z[pos_t[0]] * z[pos_t[1]]).sum(-1)
        nl = (z[neg_t[0]] * z[neg_t[1]]).sum(-1)
        logit = torch.cat([pl, nl])
        lab = torch.cat([torch.ones_like(pl), torch.zeros_like(nl)])
        loss = F.binary_cross_entropy_with_logits(logit, lab)
        loss.backward(); opt.step()
        lv = float(loss.detach())
        if lv < best - 1e-4:
            best, since = lv, 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            since += 1
            if since >= GCN_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    q_pairs = q_pos + q_neg
    yq = np.array([1] * len(q_pos) + [0] * len(q_neg))
    qe = torch.tensor([[u for u, v in q_pairs], [v for u, v in q_pairs]],
                      dtype=torch.long, device=dev)
    with torch.no_grad():
        z = model(x, ei)
        score = (z[qe[0]] * z[qe[1]]).sum(-1).cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-score))
    pred = (prob >= 0.5).astype(int)
    ba = float(balanced_accuracy_score(yq, pred))
    try:
        auc = float(roc_auc_score(yq, prob))
    except Exception:
        auc = None
    return ba, auc


# --- prompt builder -----------------------------------------------------------
def build_prompt(G_obs, data, comm, sp, k, khop, max_joint, topw, rep='opaque', texts=None):
    """One ICL prompt for a (seed,k). G_obs is the observed graph for THIS split
    (query + support positives removed). Returns (prompt_str, truth_list)."""
    what = ("each node's paper title and abstract" if rep == 'readable'
            else "each node's most active (anonymized) word features")
    TASK = (f"Each item is a PAIR of NODES (#0 and #1) in a citation network, shown "
            f"as the joint {khop}-hop neighborhood of the pair (the two nodes plus "
            f"their nearby neighbors) with {what}, followed by computed classical "
            f"link-prediction features for the pair. Decide whether an edge (a "
            f"citation link) connects #0 and #1:\n"
            f"  {TOK_POS} = LINK (an edge connects #0 and #1)\n"
            f"  {TOK_NEG} = NOLINK (no edge connects #0 and #1)\n"
            f"The query edge itself is NEVER shown in the neighborhood. Learn the "
            f"pattern from the labeled examples, then classify each query.\n"
            f"OUTPUT FORMAT: one line per query, exactly '<id> <CLASS>' where <CLASS> "
            f"is {TOK_POS} or {TOK_NEG}. No other text.")
    L = [TASK, "", "=== LABELED EXAMPLES ==="]
    for (u, v) in sp["pos_shots"][:k]:
        body, _ = verbalize_pair(G_obs, data, u, v, comm, khop, max_joint, topw, rep, texts)
        L.append(f"[{TOK_POS}]\n{body}\n")
    for (u, v) in sp["neg_shots"][:k]:
        body, _ = verbalize_pair(G_obs, data, u, v, comm, khop, max_joint, topw, rep, texts)
        L.append(f"[{TOK_NEG}]\n{body}\n")
    L.append("=== QUERIES (classify each) ===")
    truth = []
    q_pairs = [(p, TOK_POS) for p in sp["q_pos"]] + [(p, TOK_NEG) for p in sp["q_neg"]]
    # shuffle deterministically so the LLM can't infer label from position
    order = np.random.RandomState(12345).permutation(len(q_pairs))
    for qi, oi in enumerate(order):
        (u, v), tok = q_pairs[oi]
        body, _ = verbalize_pair(G_obs, data, u, v, comm, khop, max_joint, topw, rep, texts)
        L.append(f"Query {qi}:\n{body}\n")
        truth.append([qi, tok])
    return "\n".join(L), truth, [q_pairs[oi] for oi in order]


def run(dataset):
    out = f"{OUT_BASE}/{dataset}"
    os.makedirs(out, exist_ok=True)
    ds = Planetoid(ROOT, name=dataset)
    data = ds[0]
    G_full = to_networkx(data, to_undirected=True)
    G_full.remove_edges_from(nx.selfloop_edges(G_full))

    splits = make_splits(G_full, SEEDS, max(K_SHOTS), NQ)

    # readable text (rep b)
    texts, _real_names, align = load_readable(dataset)

    # observed graph + community + heuristic-logreg + GNN baselines, PER SEED.
    # The observed graph removes that seed's support+query POSITIVES (no leakage).
    # Baselines are representation-independent, computed once per (seed,k).
    hl_ba = {str(k): [] for k in K_SHOTS}; hl_auc = {str(k): [] for k in K_SHOTS}
    gn_ba = {str(k): [] for k in K_SHOTS}; gn_auc = {str(k): [] for k in K_SHOTS}
    obs_cache, comm_cache = {}, {}
    for seed in SEEDS:
        sp = splits[seed]
        removed = sp["pos_shots"] + sp["q_pos"]
        G_obs = observed_graph(G_full, removed)
        comm = _community_of(G_obs)
        obs_cache[seed] = G_obs; comm_cache[seed] = comm
        # GNN link predictor is K-independent (trained on ALL observed edges) -> once
        # per seed, replicated across K rows for table alignment.
        gba, gauc = gnn_link(data, G_obs, sp["pos_shots"], sp["neg_shots"],
                             sp["q_pos"], sp["q_neg"], None, seed)
        for k in K_SHOTS:
            ba, auc = heuristic_logreg(G_obs, comm, sp["pos_shots"], sp["neg_shots"],
                                       sp["q_pos"], sp["q_neg"], k)
            if ba is not None:
                hl_ba[str(k)].append(ba)
            if auc is not None:
                hl_auc[str(k)].append(auc)
            if gba is not None:
                gn_ba[str(k)].append(gba)
            if gauc is not None:
                gn_auc[str(k)].append(gauc)

    # prompt files PER REPRESENTATION
    rep_files = {}
    for rep in REPS:
        os.makedirs(f"{out}/{rep}/ans/opus", exist_ok=True)
        files = {}
        for seed in SEEDS:
            sp = splits[seed]
            G_obs = obs_cache[seed]; comm = comm_cache[seed]
            for k in K_SHOTS:
                prompt, truth, q_order = build_prompt(
                    G_obs, data, comm, sp, k, KHOP, MAX_JOINT, TOPW, rep=rep, texts=texts)
                fn = f"{dataset}/{rep}/seed{seed}_k{k}.txt"
                open(f"{OUT_BASE}/{fn}", 'w').write(prompt)
                files[fn] = {
                    "dataset": dataset, "rep": rep, "seed": seed, "k": k, "truth": truth,
                    "khop": KHOP, "max_joint": MAX_JOINT,
                    "query_pairs": [[int(u), int(v), tok] for (u, v), tok in q_order],
                    "support_pos": [[int(u), int(v)] for (u, v) in sp["pos_shots"][:k]],
                    "support_neg": [[int(u), int(v)] for (u, v) in sp["neg_shots"][:k]],
                    "removed_pos": [[int(u), int(v)] for (u, v) in (sp["pos_shots"] + sp["q_pos"])],
                }
        rep_files[rep] = files

    man = {"dataset": dataset, "task": "link_prediction", "seeds": SEEDS,
           "k_shots": K_SHOTS, "khop": KHOP, "max_joint": MAX_JOINT, "nq": NQ,
           "chance": 0.5, "representations": REPS, "text_alignment": align,
           "tokens": {"LINK": TOK_POS, "NOLINK": TOK_NEG},
           "link_features": LINK_FEATS,
           "files": {fn: m for fl in rep_files.values() for fn, m in fl.items()},
           "heuristic_logreg": {"bal_acc": hl_ba, "auc": hl_auc},
           "gnn_link": {"bal_acc": gn_ba, "auc": gn_auc},
           "metric": "balanced_accuracy + AUC"}
    json.dump(man, open(f"{out}/manifest.json", 'w'), indent=0)

    def fmt(d):
        return {k: f"{np.mean(v):.3f}+/-{np.std(v):.3f}" for k, v in d.items() if v}
    print(f"[{dataset}] link-pred, chance 0.500, khop={KHOP}, max_joint={MAX_JOINT}, "
          f"nq={NQ}; text={align}")
    print(f"  heuristic-logreg  bal-acc: {fmt(hl_ba)}")
    print(f"  heuristic-logreg  AUC    : {fmt(hl_auc)}")
    print(f"  GNN-link-predictor bal-acc: {fmt(gn_ba)}")
    print(f"  GNN-link-predictor AUC    : {fmt(gn_auc)}")
    n = sum(len(fl) for fl in rep_files.values())
    print(f"  wrote {n} prompt files ({len(REPS)} reps) + manifest -> {out}")
    return man


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'Cora'
    os.makedirs(OUT_BASE, exist_ok=True)
    run(dataset)
    print("run subagents (or run_qwen.py / run_opus_cli.py) on each seed*_k*.txt; "
          "score with score_edge_icl.py")
