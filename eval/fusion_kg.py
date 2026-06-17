"""Killer-app fusion smoke: does an LLM that FUSES graph STRUCTURE with WORLD
KNOWLEDGE about entities beat structure-alone AND knowledge-alone at biomedical link
prediction?

The "could-move-the-paper" experiment. A real biomedical KG carries two orthogonal
signals: TOPOLOGY (who is linked to whom) and ENTITY IDENTITY (the LLM already knows
that Doxorubicin is a chemo drug and 'breast cancer' is a malignancy). A structure-
only GNN throws identity away; a knowledge-only LLM ignores topology. If graphlex+LLM
(structure + names) beats BOTH single-signal arms, that's a capability classical
methods categorically lack. This file is the GENERATOR (no LLM) for a tiny smoke that
checks: is that gap there at all?

DATASET: Hetionet Compound-treats-Disease (CtD) — 755 edges, 387 DrugBank compounds
(readable names like Doxorubicin, Aspirin), 77 Disease-Ontology diseases ('breast
cancer', 'hypertension'). A small, real, fully human/LLM-readable biomedical KG.
Loaded from kg_data/hetionet/CtD.tsv (compound_id, compound_name, disease_id,
disease_name). Bipartite: compounds on one side, diseases on the other; the single
relation is "treats".

TASK: binary link prediction — "does edge (compound, TREATS, disease) exist?" Balanced
held-out positives + sampled negatives (true compound-disease NON-edges, both endpoints
with degree>0 so neighborhoods exist). CLASS1 = TREATS, CLASS0 = NO-TREAT. Few-shot K
class-balanced labeled examples. Metric: balanced accuracy (_common.bal_acc) + AUC.

LEAKAGE: the positive QUERY edges and positive SUPPORT edges are REMOVED (undirected
pair set, same discipline as edge_icl.py / kg_icl.py) from the observed graph used to
(a) build the verbalized neighborhoods, and (b) train/feature the non-LLM baseline.

THE THREE ARMS (prompt CONDITIONS over the SAME query edges; identical truth):
  1. knowledge  — show entity NAMES, NO structure: "Is <drug> a treatment for
     <disease>?" plus K labeled name-only examples. Pure parametric recall; ALSO the
     memorization gauge (if near-ceiling alone -> task is memorized -> temporal holdout
     needed).
  2. anon       — graphlex verbalizes the query edge's joint 1-hop neighborhood
     (reusing the edge_icl/kg_icl joint-neighborhood machinery) but ALL entities are
     ANONYMIZED (C001, D001, ...). Topology without identity. Also the anonymization
     control.
  3. fusion     — graphlex neighborhood + REAL entity names. Both signals.

NON-LLM REFERENCE: a structural link-pred baseline — classical heuristics
(common-neighbors / Adamic-Adar / Jaccard / resource-allocation / preferential-
attachment / shortest-path, computed on the leakage-stripped bipartite graph) fed to
logreg on the K support pairs. This is the structure-only classical bar.

graphlex CORE IS NOT MODIFIED: we call facts()/verbalize(focus='structure') on a
relabeled neighborhood and APPEND a readable typed-context line, exactly as kg_icl.py /
edge_icl.py do.

Outputs (run_opus_cli.py / run_qwen.py compatible):
  /home/scratch/bench_out/fusion_kg/<DS>/{knowledge,anon,fusion}/seed*_k*.txt
  /home/scratch/bench_out/fusion_kg/<DS>/manifest.json  (truth + baseline)

Run: /home/scratch/fmsn-dev/.venv/bin/python eval/fusion_kg.py [hetionet]
     env SMOKE=1 -> tiny grid (the budgeted smoke). Calls NO LLM.
"""
import os, sys, json
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
from graphlex import facts, verbalize

KG_BASE = '/home/scratch/kg_data'
OUT_BASE = '/home/scratch/bench_out/fusion_kg'
CONDITIONS = ['knowledge', 'anon', 'fusion']
SEEDS = [11, 22, 33]            # >=2 seeds
K_SHOTS = [1, 3]               # class-balanced shots PER CLASS
KHOP = 1                        # joint-neighborhood radius per endpoint
MAX_JOINT = 16                  # cap joint-neighborhood node count
MAX_CTX_EDGES = 16              # cap typed-context edges shown per pair
NQ = 40                         # query pairs (NQ/2 positive + NQ/2 negative)
TOK_POS, TOK_NEG = 'CLASS1', 'CLASS0'   # TREATS / NO-TREAT (reuse _common.parse_ans)

# SMOKE: the budgeted run. <=12 query items, k in {1,3}, 2 seeds.
if os.environ.get('SMOKE'):
    SEEDS, K_SHOTS, NQ = [11, 22], [1, 3], 12
if os.environ.get('FULL'):
    SEEDS, K_SHOTS, NQ = [11, 22, 33, 44, 55], [1, 3, 5], 60


# --- KG loading ---------------------------------------------------------------
def load_ctd():
    """Load Hetionet Compound-treats-Disease from kg_data/hetionet/CtD.tsv. Returns
    (edges, c_name, d_name) where edges = list of (compound_id, disease_id) and
    c_name/d_name map id -> readable name. Bipartite: compound side vs disease side."""
    path = f"{KG_BASE}/hetionet/CtD.tsv"
    edges, c_name, d_name = [], {}, {}
    with open(path) as f:
        next(f)
        for ln in f:
            p = ln.rstrip('\n').split('\t')
            if len(p) != 4:
                continue
            cid, cnm, did, dnm = p
            edges.append((cid, did))
            c_name[cid] = cnm
            d_name[did] = dnm
    return edges, c_name, d_name


# --- observed graph (leakage control, undirected pair discipline) -------------
def build_nx(edges, removed):
    """Undirected bipartite networkx graph of all TREATS edges EXCEPT removed pairs
    (a set of frozenset({cid,did})). Same leakage discipline as kg_icl.observed_triples."""
    G = nx.Graph()
    G.add_nodes_from({c for c, d in edges})
    G.add_nodes_from({d for c, d in edges})
    for (c, d) in edges:
        if frozenset((c, d)) in removed:
            continue
        G.add_edge(c, d)
    return G


# --- joint 1-hop neighborhood (untyped skeleton for graphlex) -----------------
def joint_nx(G, h, t, khop, max_joint):
    """Untyped joint k-hop neighborhood of (h,t) as a relabeled networkx graph, capped
    to <=max_joint nodes; ids 0,1 = h,t. The h-t edge is stripped so the query is not
    revealed structurally. Returns (H, glob) with glob[0]=h, glob[1]=t. (Same shape as
    kg_icl.joint_nx.)"""
    def khop_nodes(src):
        seen = {src}; frontier = {src}
        for _ in range(khop):
            nxt = set()
            for x in frontier:
                if x in G:
                    nxt |= set(G.neighbors(x))
            nxt -= seen; seen |= nxt; frontier = nxt
        return seen
    nodes = (khop_nodes(h) | khop_nodes(t)) if (h in G or t in G) else set()
    nodes.discard(h); nodes.discard(t)
    rest = list(nodes)[:max_joint - 2]
    glob = [h, t] + rest
    idx = {g: i for i, g in enumerate(glob)}
    gset = set(glob)
    H = nx.Graph()
    H.add_nodes_from(range(len(glob)))
    for a in glob:
        if a not in G:
            continue
        for b in G.neighbors(a):
            if b in gset and a != b:
                ai, bi = idx[a], idx[b]
                if {ai, bi} == {0, 1}:      # never reveal the query pair's link
                    continue
                H.add_edge(ai, bi)
    return H, glob


# --- typed-context edge list (appended; graphlex core untouched) --------------
def context_edges(G, h, t, name_of, max_edges):
    """Readable list of observed TREATS edges incident to h or t (the query pair's own
    edge excluded by construction — it is not in the observed graph). name_of maps an
    id to the label to print (real name for fusion; anon token for anon — and for anon
    name_of MUST cover EVERY entity it is asked about, including context-only neighbors,
    so no raw coded id ever leaks identity). Parallel to kg_icl.typed_context /
    edge_icl's appended feature line."""
    seen = set(); lines = []
    for ent in (h, t):
        if ent not in G:
            continue
        for nb in G.neighbors(ent):
            key = frozenset((ent, nb))
            if key in seen:
                continue
            seen.add(key)
            # orient compound -[treats]-> disease using which side each id is
            a, b = (ent, nb)
            lines.append(f"({name_of(a)}) -[treats]- ({name_of(b)})")
            if len(lines) >= max_edges:
                break
        if len(lines) >= max_edges:
            break
    if not lines:
        return "Known treatment edges around the pair (excluding the query): (none observed)"
    return ("Known treatment edges around the pair (excluding the query):\n    "
            + "\n    ".join(lines))


# --- per-condition verbalization of one query pair ----------------------------
def anon_label(glob):
    """Anonymized token per global id by position in the neighborhood: A00=head,
    A01=tail, A02.. for neighbors. Identity-free, topology-preserving."""
    return {g: f"A{i:02d}" for i, g in enumerate(glob)}


def verbalize_struct(G, h, t, names, anon, khop, max_joint, max_ctx):
    """Joint-neighborhood structural verbalization (graphlex) + a typed-context edge
    list. If anon=True, every entity is an Axx token (topology, no identity); else real
    names. Returns the body string. #0=h (compound), #1=t (disease)."""
    H, glob = joint_nx(G, h, t, khop, max_joint)
    struct = verbalize(facts(H), focus='structure')
    if anon:
        amap = anon_label(glob)        # A00=head, A01=tail, A02.. = capped neighbors

        def name_of(g):
            """Anon token; mint a fresh one for any context-only entity beyond the
            capped neighborhood so NO raw coded id ever leaks identity in the anon arm."""
            if g not in amap:
                amap[g] = f"A{len(amap):02d}"
            return amap[g]
        head_lbl, tail_lbl = name_of(h), name_of(t)
        ent_kind_c, ent_kind_d = "compound", "disease"
    else:
        name_of = lambda g: names.get(g, g)
        head_lbl, tail_lbl = name_of(h), name_of(t)
        ent_kind_c, ent_kind_d = "compound (drug)", "disease"
    ctx = context_edges(G, h, t, name_of, max_ctx)
    return (f"HEAD = #0 = {ent_kind_c} '{head_lbl}'  ,  TAIL = #1 = {ent_kind_d} "
            f"'{tail_lbl}'  (predict whether HEAD treats TAIL).\n"
            f"Untyped {khop}-hop neighborhood skeleton of the pair: {struct}\n"
            f"{ctx}")


def verbalize_knowledge(c, d, c_name, d_name):
    """Knowledge-only body: just the entity NAMES, NO structure/neighborhood."""
    return (f"Compound (drug): '{c_name.get(c, c)}'\n"
            f"Disease: '{d_name.get(d, d)}'\n"
            f"Question: is this compound a treatment for this disease?")


def verbalize_pair(cond, G, c, d, c_name, d_name, names, khop, max_joint, max_ctx):
    """Dispatch verbalization for one (compound, disease) query pair by condition."""
    if cond == 'knowledge':
        return verbalize_knowledge(c, d, c_name, d_name)
    return verbalize_struct(G, c, d, names, anon=(cond == 'anon'),
                            khop=khop, max_joint=max_joint, max_ctx=max_ctx)


# --- balanced pos/neg sampling ------------------------------------------------
def sample_pairs(edges, comps, diss, eset, n_pos, n_neg, rng):
    """n_pos positive TREATS edges + n_neg true NON-edges (compound, disease) with both
    endpoints already incident to >=1 edge (so neighborhoods are non-empty). We keep the
    (compound, disease) ORDER (positives drawn from the ordered `edges` list, NOT from
    the unordered frozenset `eset`, so orientation is never scrambled). Returns (pos,
    neg) lists of (cid, did)."""
    edge_list = list({(c, d) for c, d in edges})    # ordered (compound, disease) pairs
    rng.shuffle(edge_list)
    pos = edge_list[:n_pos]
    deg_comp = [c for c in comps]
    deg_dis = [d for d in diss]
    neg, seen = [], set()
    tries = 0
    while len(neg) < n_neg and tries < n_neg * 5000:
        tries += 1
        c = deg_comp[rng.randint(len(deg_comp))]
        d = deg_dis[rng.randint(len(deg_dis))]
        if frozenset((c, d)) in eset or (c, d) in seen:
            continue
        seen.add((c, d)); neg.append((c, d))
    return pos, neg


def make_splits(edges, c_name, d_name, seeds, k_max, nq):
    """Per seed: k_max pos + k_max neg SUPPORT pairs, nq/2 pos + nq/2 neg QUERY pairs,
    all disjoint pairs. Only compounds/diseases with degree>0 are used as endpoints."""
    comps = sorted({c for c, d in edges})
    diss = sorted({d for c, d in edges})
    eset = {frozenset((c, d)) for c, d in edges}
    nqh = nq // 2
    out = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        pos, neg = sample_pairs(edges, comps, diss, eset, k_max + nqh, k_max + nqh, rng)
        out[seed] = {
            "pos_shots": pos[:k_max], "neg_shots": neg[:k_max],
            "q_pos": pos[k_max:k_max + nqh], "q_neg": neg[k_max:k_max + nqh],
        }
    return out


# --- non-LLM structural baseline (classical heuristics + logreg) --------------
LINK_FEATS = ['common_neighbors', 'jaccard', 'adamic_adar', 'resource_allocation',
              'preferential_attachment', 'shortest_path']


def link_features(G, u, v):
    """Classical link-prediction features for (u,v) on the leakage-stripped graph.
    Bipartite, so 1-hop common-neighbors are 0; the signal is in 2-hop heuristics
    (Adamic-Adar / resource-allocation over shared neighbors) and path length."""
    nu = set(G.neighbors(u)) if u in G else set()
    nv = set(G.neighbors(v)) if v in G else set()
    cn = len(nu & nv)
    def _idx(fn):
        try:
            return next(iter(fn(G, [(u, v)])))[2]
        except Exception:
            return 0.0
    jac = _idx(nx.jaccard_coefficient)
    aa = _idx(nx.adamic_adar_index)
    ra = _idx(nx.resource_allocation_index)
    try:
        pa = next(iter(nx.preferential_attachment(G, [(u, v)])))[2]
    except Exception:
        pa = float(len(nu) * len(nv))
    try:
        sp = nx.shortest_path_length(G, u, v)
    except Exception:
        sp = -1
    return {'common_neighbors': float(cn), 'jaccard': float(jac),
            'adamic_adar': float(aa), 'resource_allocation': float(ra),
            'preferential_attachment': float(pa), 'shortest_path': float(sp)}


def feat_vec(d):
    return [d[k] for k in LINK_FEATS]


def heuristic_logreg(G, pos_shots, neg_shots, q_pos, q_neg, k):
    """Structure-only classical bar: logreg on the link-feature vectors of K pos + K
    neg support pairs, evaluated on the balanced query pairs. All features on the
    leakage-stripped observed graph. Returns (balanced_acc, auc)."""
    tr = pos_shots[:k] + neg_shots[:k]
    ytr = [1] * k + [0] * k
    if len(set(ytr)) < 2:
        return None, None
    Xtr = np.array([feat_vec(link_features(G, u, v)) for (u, v) in tr])
    qp = q_pos + q_neg
    yq = np.array([1] * len(q_pos) + [0] * len(q_neg))
    Xq = np.array([feat_vec(link_features(G, u, v)) for (u, v) in qp])
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


# --- prompt builder -----------------------------------------------------------
def build_prompt(cond, G, sp, k, c_name, d_name, khop, max_joint, max_ctx):
    """One ICL prompt for a (condition, seed, k). Returns (prompt, truth, q_order).
    truth = [[qid, TOKEN], ...]. names = id->real name (used by anon/fusion; ignored by
    knowledge)."""
    names = {**c_name, **d_name}
    if cond == 'knowledge':
        signal = ("You are shown ONLY the entity NAMES — no graph, no neighborhood. "
                  "Use your knowledge of drugs and diseases.")
    elif cond == 'anon':
        signal = ("You are shown the ANONYMIZED local graph structure around the pair "
                  "(entities are opaque tokens A00, A01, ...; the query link is never "
                  "shown). Use TOPOLOGY only — you do not know which drug or disease "
                  "these are.")
    else:  # fusion
        signal = ("You are shown BOTH the local graph structure around the pair AND the "
                  "real entity NAMES. Combine topology with your knowledge of drugs and "
                  "diseases (the query link is never shown).")
    TASK = (
        "Each item is an ordered pair (HEAD = a compound/drug, TAIL = a disease) from a "
        "biomedical knowledge graph. Your job is binary LINK PREDICTION: does the "
        "compound TREAT the disease?\n"
        f"  {TOK_POS} = TREATS (the compound is an indicated treatment for the disease)\n"
        f"  {TOK_NEG} = NO-TREAT (it is not)\n"
        f"{signal}\n"
        "Learn the decision from the labeled examples, then classify each query.\n"
        f"OUTPUT FORMAT: one line per query, exactly '<id> <CLASS>' where <CLASS> is "
        f"{TOK_POS} or {TOK_NEG}. No other text.")
    L = [TASK, "", "=== LABELED EXAMPLES ==="]
    for (c, d) in sp["pos_shots"][:k]:
        body = verbalize_pair(cond, G, c, d, c_name, d_name, names, khop, max_joint, max_ctx)
        L.append(f"[{TOK_POS}]\n{body}\n")
    for (c, d) in sp["neg_shots"][:k]:
        body = verbalize_pair(cond, G, c, d, c_name, d_name, names, khop, max_joint, max_ctx)
        L.append(f"[{TOK_NEG}]\n{body}\n")
    L.append("=== QUERIES (classify each) ===")
    q_pairs = [(p, TOK_POS) for p in sp["q_pos"]] + [(p, TOK_NEG) for p in sp["q_neg"]]
    order = np.random.RandomState(12345).permutation(len(q_pairs))
    truth, q_order = [], []
    for qi, oi in enumerate(order):
        (c, d), tok = q_pairs[oi]
        body = verbalize_pair(cond, G, c, d, c_name, d_name, names, khop, max_joint, max_ctx)
        L.append(f"Query {qi}:\n{body}\n")
        truth.append([qi, tok])
        q_order.append(((c, d), tok))
    return "\n".join(L), truth, q_order


def run(dataset='hetionet'):
    out = f"{OUT_BASE}/{dataset}"
    os.makedirs(out, exist_ok=True)
    edges, c_name, d_name = load_ctd()
    n_comp = len({c for c, d in edges}); n_dis = len({d for c, d in edges})

    splits = make_splits(edges, c_name, d_name, SEEDS, max(K_SHOTS), NQ)

    # observed graph + non-LLM baseline, per seed (leakage-stripped per split).
    hl_ba = {str(k): [] for k in K_SHOTS}; hl_auc = {str(k): [] for k in K_SHOTS}
    obs_cache = {}
    for seed in SEEDS:
        sp = splits[seed]
        removed = {frozenset((c, d)) for (c, d) in sp["pos_shots"] + sp["q_pos"]}
        G = build_nx(edges, removed)
        obs_cache[seed] = G
        for k in K_SHOTS:
            ba, auc = heuristic_logreg(G, sp["pos_shots"], sp["neg_shots"],
                                       sp["q_pos"], sp["q_neg"], k)
            if ba is not None:
                hl_ba[str(k)].append(ba)
            if auc is not None:
                hl_auc[str(k)].append(auc)

    # prompt files PER CONDITION (same query edges across conditions)
    files = {}
    for cond in CONDITIONS:
        os.makedirs(f"{out}/{cond}/ans/opus", exist_ok=True)
        os.makedirs(f"{out}/{cond}/ans/qwen", exist_ok=True)
        for seed in SEEDS:
            sp = splits[seed]; G = obs_cache[seed]
            for k in K_SHOTS:
                prompt, truth, q_order = build_prompt(
                    cond, G, sp, k, c_name, d_name, KHOP, MAX_JOINT, MAX_CTX_EDGES)
                fn = f"{dataset}/{cond}/seed{seed}_k{k}.txt"
                with open(f"{OUT_BASE}/{fn}", 'w') as fh:
                    fh.write(prompt)
                files[fn] = {
                    "dataset": dataset, "condition": cond, "seed": seed, "k": k,
                    "truth": truth, "khop": KHOP, "max_joint": MAX_JOINT,
                    "query_pairs": [[c, d, tok] for (c, d), tok in q_order],
                }

    man = {"dataset": dataset, "task": "treats_link_prediction",
           "relation": "Compound-treats-Disease (Hetionet)",
           "conditions": CONDITIONS, "seeds": SEEDS, "k_shots": K_SHOTS,
           "khop": KHOP, "max_joint": MAX_JOINT, "nq": NQ,
           "n_compounds": n_comp, "n_diseases": n_dis, "n_edges": len(edges),
           "chance": 0.5, "tokens": {"TREATS": TOK_POS, "NO_TREAT": TOK_NEG},
           "link_features": LINK_FEATS,
           "files": files,
           "structural_logreg": {"bal_acc": hl_ba, "auc": hl_auc},
           "metric": "balanced_accuracy (primary) + AUC"}
    with open(f"{out}/manifest.json", 'w') as fh:
        json.dump(man, fh, indent=0)

    def fmt(d):
        return {k: f"{np.mean(v):.3f}+/-{np.std(v):.3f}" for k, v in d.items() if v}
    print(f"[{dataset}] CtD link-pred: compounds={n_comp}, diseases={n_dis}, "
          f"edges={len(edges)}, chance=0.500, khop={KHOP}, nq={NQ}, seeds={SEEDS}")
    print(f"  structural-logreg  bal-acc: {fmt(hl_ba)}")
    print(f"  structural-logreg  AUC    : {fmt(hl_auc)}")
    print(f"  wrote {len(files)} prompt files ({len(CONDITIONS)} conditions) + manifest -> {out}")
    return man


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'hetionet'
    os.makedirs(OUT_BASE, exist_ok=True)
    run(dataset)
    print("run run_qwen.py / run_opus_cli.py on each {knowledge,anon,fusion}/seed*_k*.txt; "
          "score with score_fusion_kg.py")
