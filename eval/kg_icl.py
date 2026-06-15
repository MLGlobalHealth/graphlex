"""Knowledge-graph RELATION-PREDICTION few-shot ICL track: graphlex+LLM vs a small
KG-embedding baseline (DistMult, numpy) vs a frequency/prior baseline, on a small
multi-relational KG (UMLS).

FOURTH task granularity, and the MULTI-RELATIONAL variant of the edge track. The
whole-graph (sweep.py), node (node_icl.py), and edge/link (edge_icl.py) tracks all
ask yes/no-or-class questions about UNTYPED structure. Here the edges have TYPES: the
KG is a set of (head, relation, tail) triples over a relation vocabulary. The task is
RELATION PREDICTION: given a head h and tail t that ARE linked, predict which
relation r connects them, as a classification over the relation vocabulary. This is
ULTRA's home turf (arXiv 2310.04562) and extends the granularity-flexibility story to
TYPED edges, where graphlex+LLM still spans the task in one results table while each
specialist FM is locked to its granularity.

Closely parallels edge_icl.py — read that first. The differences are:
  * a "query" is an ORDERED pair (h, t) KNOWN to be linked; label = the relation r
    (a MULTI-CLASS target over the relation vocabulary R), not a binary LINK/NOLINK.
  * relations are mapped to CLASS tokens (R00..) so _common.parse_ans + the existing
    drivers + _common.bal_acc work UNCHANGED. The prompt lists the candidate
    relations (CLASS token -> readable relation name) up front so the LLM picks one.
  * VERBALIZATION (the graphlex angle): for a pair (h,t) we verbalize the UNTYPED
    structural skeleton of their joint 1-hop neighborhood via graphlex facts()/
    verbalize(focus='structure'), PLUS — because graphlex core renders untyped
    structure — we APPEND a readable TYPED-TRIPLES context line listing the observed
    (h, rel, x) and (x, rel, t) triples around the pair (the query triple itself is
    NEVER shown). This is exactly how edge_icl.py appends its computed link-feature
    line: graphlex core is NOT modified.

LEAKAGE: the query triple (h, ?, t) and the support query triples are REMOVED from
the observed graph used to (a) build the verbalized neighborhood, (b) list the typed
context, and (c) train the KG-embedding baseline. So no method sees the relation it
must predict. (Other relations between the same (h,t) pair, if any, are also stripped
so the answer can't be read off a parallel edge.)

BASELINES (run, no LLM):
  1. DistMult (numpy, tiny epoch budget) — a standard KG-embedding model; scored as
     relation prediction by ranking r* = argmax_r <e_h, w_r, e_t> over the relation
     vocabulary for each query pair. Balanced accuracy + Hits@1 + MRR.
  2. Frequency/prior — predict the globally most-common relation (a single class);
     balanced accuracy is the floor. Also a smarter prior: most-common relation given
     (degree-bucketed) endpoints is overkill for the smoke; we keep global most-common.

ULTRA (the FM foil) is ENV-PENDING — see KG_TRACK_PLAN.md for the zero-shot inductive
relation-prediction slot-in (same query pairs, same balanced-accuracy metric). Not
installed here (needs a CUDA env on clpc35).

Metric: BALANCED accuracy over the relation classes (primary, reuses _common.bal_acc)
+ Hits@1 / MRR for the ranking baselines. Mean over seeds, into manifest.json.

Run:  /home/scratch/fmsn-dev/.venv/bin/python eval/kg_icl.py [DATASET]
      DATASET in {UMLS, Nations, Kinship}; default UMLS. Env SMOKE=1 -> tiny grid.
      Calls NO LLM.
"""
import os, sys, json
import numpy as np
import networkx as nx

sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
from graphlex import facts, verbalize

KG_BASE = '/home/scratch/kg_data'
OUT_BASE = '/home/scratch/bench_out/kg_icl'
SEEDS = [11, 22, 33]            # >=3 seeds
K_SHOTS = [1, 3]               # labeled (h,t -> r) example pairs PER QUERY BLOCK
KHOP = 1                        # joint-neighborhood radius per endpoint
MAX_JOINT = 16                  # cap joint-neighborhood node count
MAX_CTX_TRIPLES = 24            # cap typed-context triples shown per pair
NQ = 20                         # query pairs (sampled from the test split)
N_CANDREL = None                # None = full relation vocab as candidates
REPS = ['readable']             # KG entities/relations ARE human-readable -> one rep

# DistMult baseline hyperparameters (tiny, CPU, numpy). lr/epochs/dim picked by a
# quick sweep on UMLS train-triple relation recovery (lr=0.05, dim=64 was the knee:
# bal-acc ~0.5, Hits@1 ~0.48, MRR ~0.65 vs ~0.022 chance / ~0.11 freq-prior floor).
DM_DIM = 64
DM_EPOCHS = 300
DM_LR = 0.05
DM_REG = 1e-3
DM_NEG = 4                       # negative samples per positive

if os.environ.get('SMOKE'):
    SEEDS, K_SHOTS, NQ = [11, 22], [1, 3], 12


# --- KG loading ---------------------------------------------------------------
def load_kg(dataset):
    """Load (train, valid, test) triple lists + entity/relation vocabularies from
    tab-separated h\\tr\\tt files under KG_BASE/<dataset>/. Returns
    (triples_all, train, test, ent2id, rel2id, id2rel) with triples as (h_id,r_id,t_id)."""
    def read(split):
        out = []
        path = f"{KG_BASE}/{dataset}/{split}.txt"
        if not os.path.exists(path):
            return out
        for ln in open(path):
            p = ln.rstrip('\n').split('\t')
            if len(p) != 3:
                continue
            out.append(tuple(p))           # (h, r, t) as strings
        return out
    train_s = read('train')
    valid_s = read('valid')
    test_s = read('test')
    all_s = train_s + valid_s + test_s
    ents = sorted({h for h, r, t in all_s} | {t for h, r, t in all_s})
    rels = sorted({r for h, r, t in all_s})
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}
    id2rel = {i: r for r, i in rel2id.items()}
    id2ent = {i: e for e, i in ent2id.items()}
    enc = lambda S: [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in S]
    return (enc(all_s), enc(train_s), enc(test_s), ent2id, rel2id, id2rel, id2ent)


# relation CLASS tokens: R00, R01, ... so _common.parse_ans / bal_acc work unchanged.
def rel_token(rid):
    return f"R{rid:02d}"


# --- observed graph (leakage control) -----------------------------------------
def observed_triples(all_triples, removed_pairs):
    """All triples EXCEPT any with (head,tail) in removed_pairs (an undirected pair
    set), so neither the query relation nor a parallel relation on the same pair leaks.
    removed_pairs: set of frozenset({h,t})."""
    out = []
    for (h, r, t) in all_triples:
        if frozenset((h, t)) in removed_pairs:
            continue
        out.append((h, r, t))
    return out


def build_nx(obs_triples):
    """Untyped directed-as-undirected skeleton networkx graph of the observed triples
    (for graphlex structural verbalization) + an adjacency dict of typed triples per
    entity for the typed-context line."""
    G = nx.Graph()
    typed = {}                              # entity -> list of (h,r,t) incident triples
    for (h, r, t) in obs_triples:
        if h != t:
            G.add_edge(h, t)
        typed.setdefault(h, []).append((h, r, t))
        typed.setdefault(t, []).append((h, r, t))
    return G, typed


# --- joint 1-hop neighborhood (untyped skeleton for graphlex) -----------------
def joint_nx(G, h, t, khop, max_joint):
    """Untyped joint k-hop neighborhood of (h,t) as a relabeled networkx graph, capped
    to <=max_joint nodes; ids 0,1 = h,t. (The h-t edge, if present in the skeleton, is
    stripped so the query is not revealed structurally.) Returns (H, glob)."""
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


# --- typed-triples context line (appended; graphlex core untouched) -----------
def typed_context(typed, h, t, id2ent, id2rel, exclude_pair, max_triples):
    """Readable list of observed typed triples incident to h or t (the query pair's
    own triples excluded), parallel to edge_icl's appended link-feature line."""
    seen = set(); lines = []
    for ent, tag in ((h, 'HEAD'), (t, 'TAIL')):
        for (hh, rr, tt) in typed.get(ent, []):
            if frozenset((hh, tt)) == exclude_pair:
                continue
            key = (hh, rr, tt)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"({id2ent[hh]}) -[{id2rel[rr]}]-> ({id2ent[tt]})")
            if len(lines) >= max_triples:
                break
        if len(lines) >= max_triples:
            break
    if not lines:
        return "Known typed triples around the pair (excluding the query): (none observed)"
    return ("Known typed triples around the pair (excluding the query):\n    "
            + "\n    ".join(lines))


def verbalize_pair(G, typed, h, t, id2ent, id2rel, khop, max_joint, max_ctx):
    """Verbalize a query pair: untyped structural skeleton of the joint neighborhood
    (graphlex) + the readable typed-triples context (appended). #0=h, #1=t."""
    H, glob = joint_nx(G, h, t, khop, max_joint)
    struct = verbalize(facts(H), focus='structure')
    ctx = typed_context(typed, h, t, id2ent, id2rel, frozenset((h, t)), max_ctx)
    return (f"HEAD = #0 = '{id2ent[h]}'  ,  TAIL = #1 = '{id2ent[t]}'  "
            f"(predict the relation r such that HEAD -[r]-> TAIL).\n"
            f"Untyped {khop}-hop neighborhood skeleton of the pair: {struct}\n"
            f"{ctx}")


# --- DistMult KG-embedding baseline (numpy) -----------------------------------
def train_distmult(obs_triples, n_ent, n_rel, seed,
                   dim=DM_DIM, epochs=DM_EPOCHS, lr=DM_LR, reg=DM_REG, neg=DM_NEG):
    """Tiny numpy DistMult: score(h,r,t) = sum(e_h * w_r * e_t). Margin-free
    logistic loss with random corrupted-tail negatives. Returns (E, W)."""
    rng = np.random.RandomState(seed)
    E = rng.normal(0, 0.1, size=(n_ent, dim))
    W = rng.normal(0, 0.1, size=(n_rel, dim))
    tri = np.array(obs_triples)
    if len(tri) == 0:
        return E, W
    for ep in range(epochs):
        rng.shuffle(tri)
        for (h, r, t) in tri:
            eh, wr, et = E[h], W[r], E[t]
            # positive
            sp = float(np.dot(eh * wr, et))
            gp = 1.0 / (1.0 + np.exp(-sp)) - 1.0      # dloss/dscore, label 1
            # negatives (corrupt tail)
            negs = rng.randint(0, n_ent, size=neg)
            grad_eh = gp * (wr * et)
            grad_wr = gp * (eh * et)
            grad_et = gp * (wr * eh)
            for tn in negs:
                en = E[tn]
                sn = float(np.dot(eh * wr, en))
                gn = 1.0 / (1.0 + np.exp(-sn))         # label 0
                grad_eh += gn * (wr * en)
                grad_wr += gn * (eh * en)
                E[tn] -= lr * (gn * (eh * wr) + reg * en)
            E[h] -= lr * (grad_eh + reg * eh)
            W[r] -= lr * (grad_wr + reg * wr)
            E[t] -= lr * (grad_et + reg * et)
    return E, W


def distmult_relpred(E, W, query_pairs):
    """For each (h,t) query pair, rank relations by DistMult score; return
    (pred_rel_ids, ranks_of_truth_unknown_here). Here we only need predicted argmax
    per pair (truth handled by caller). Returns list of full score rankings (n_rel
    relation ids sorted best-first) per pair."""
    rankings = []
    for (h, t) in query_pairs:
        # score over all relations: <E_h * W_r, E_t> = sum_d E_h[d]*W[r,d]*E_t[d]
        scores = (W * (E[h] * E[t])[None, :]).sum(1)   # (n_rel,)
        order = np.argsort(-scores)
        rankings.append(order.tolist())
    return rankings


def ranking_metrics(rankings, truths):
    """Balanced accuracy of argmax prediction + Hits@1 + MRR, given per-pair relation
    rankings (best-first) and the true relation id per pair."""
    preds = [r[0] for r in rankings]
    # balanced acc via _common.bal_acc-equivalent: macro recall over true classes
    by = {}
    for i, ti in enumerate(truths):
        by.setdefault(ti, []).append(i)
    recs = [np.mean([preds[i] == lab for i in ids]) for lab, ids in by.items()]
    ba = float(np.mean(recs)) if recs else None
    hits1 = float(np.mean([rankings[i][0] == truths[i] for i in range(len(truths))]))
    mrr = float(np.mean([1.0 / (rankings[i].index(truths[i]) + 1) for i in range(len(truths))]))
    return ba, hits1, mrr


def freq_prior_relpred(obs_triples, truths):
    """Most-common-relation prior: predict the single globally most-frequent relation
    for every query. Balanced accuracy (will be ~1/n_classes_present)."""
    if not obs_triples:
        return None
    rels = [r for (_, r, _) in obs_triples]
    most = max(set(rels), key=rels.count)
    by = {}
    for i, ti in enumerate(truths):
        by.setdefault(ti, []).append(i)
    recs = [np.mean([most == lab for _ in ids]) for lab, ids in by.items()]
    return float(np.mean(recs)) if recs else None


# --- splits -------------------------------------------------------------------
def make_splits(test_triples, all_triples, seeds, k_max, nq):
    """Per seed: nq query triples (sampled from test) + k_max support triples (sampled
    from the rest), all distinct pairs. Returns {seed: {support, query}} with triples
    as (h,r,t). De-dups so a (h,t) pair is not both support and query."""
    out = {}
    pool_test = list({(h, r, t) for (h, r, t) in test_triples})
    pool_supp = list({(h, r, t) for (h, r, t) in all_triples})
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(pool_test))
        query, qpairs = [], set()
        for i in idx:
            h, r, t = pool_test[i]
            fs = frozenset((h, t))
            if fs in qpairs:
                continue
            qpairs.add(fs); query.append(pool_test[i])
            if len(query) >= nq:
                break
        sidx = rng.permutation(len(pool_supp))
        support = []
        for i in sidx:
            h, r, t = pool_supp[i]
            fs = frozenset((h, t))
            if fs in qpairs:
                continue
            support.append(pool_supp[i])
            if len(support) >= k_max:
                break
        out[seed] = {"support": support, "query": query}
    return out


# --- prompt builder -----------------------------------------------------------
def build_prompt(G, typed, support, query, k, id2ent, id2rel, rel2id,
                 khop, max_joint, max_ctx):
    """One ICL relation-prediction prompt for a (seed,k). Returns (prompt, truth_list,
    query_order). truth_list = [[qid, REL_TOKEN], ...]."""
    # candidate relation menu (full vocab)
    rel_ids = sorted(id2rel.keys())
    menu = "\n".join(f"  {rel_token(rid)} = {id2rel[rid]}" for rid in rel_ids)
    TASK = (
        "Each item is an ORDERED PAIR of entities (HEAD = #0, TAIL = #1) in a "
        "knowledge graph that ARE linked by exactly one relation. Your job is "
        "RELATION PREDICTION: choose which relation r connects HEAD -[r]-> TAIL, from "
        "the candidate relations below. You are shown the untyped structural skeleton "
        "of the pair's local neighborhood plus the known TYPED triples around the two "
        "entities (the query triple itself is never shown). Learn the pattern from the "
        "labeled examples, then classify each query.\n\n"
        "CANDIDATE RELATIONS (answer with the CLASS token on the left):\n"
        f"{menu}\n\n"
        "OUTPUT FORMAT: one line per query, exactly '<id> <CLASS>' where <CLASS> is "
        "one of the relation tokens above (e.g. 'R03'). No other text.")
    L = [TASK, "", "=== LABELED EXAMPLES ==="]
    for (h, r, t) in support[:k]:
        body = verbalize_pair(G, typed, h, t, id2ent, id2rel, khop, max_joint, max_ctx)
        L.append(f"[{rel_token(r)}]\n{body}\n")
    L.append("=== QUERIES (classify each) ===")
    truth = []
    order = np.random.RandomState(12345).permutation(len(query))
    q_order = []
    for qi, oi in enumerate(order):
        h, r, t = query[oi]
        body = verbalize_pair(G, typed, h, t, id2ent, id2rel, khop, max_joint, max_ctx)
        L.append(f"Query {qi}:\n{body}\n")
        truth.append([qi, rel_token(r)])
        q_order.append((h, r, t))
    return "\n".join(L), truth, q_order


def run(dataset):
    out = f"{OUT_BASE}/{dataset}"
    os.makedirs(out, exist_ok=True)
    (all_tr, train_tr, test_tr, ent2id, rel2id, id2rel, id2ent) = load_kg(dataset)
    n_ent, n_rel = len(ent2id), len(rel2id)

    splits = make_splits(test_tr, all_tr, SEEDS, max(K_SHOTS), NQ)

    # baselines per seed: observed graph removes that seed's query + support pairs.
    dm_ba = []; dm_h1 = []; dm_mrr = []; fp_ba = []
    obs_cache = {}
    for seed in SEEDS:
        sp = splits[seed]
        removed = {frozenset((h, t)) for (h, r, t) in sp["query"] + sp["support"]}
        obs = observed_triples(all_tr, removed)
        G, typed = build_nx(obs)
        obs_cache[seed] = (obs, G, typed)
        q_pairs = [(h, t) for (h, r, t) in sp["query"]]
        q_truth = [r for (h, r, t) in sp["query"]]
        # DistMult (trained on observed triples only)
        E, W = train_distmult(obs, n_ent, n_rel, seed)
        rankings = distmult_relpred(E, W, q_pairs)
        ba, h1, mrr = ranking_metrics(rankings, q_truth)
        dm_ba.append(ba); dm_h1.append(h1); dm_mrr.append(mrr)
        # frequency prior
        fp = freq_prior_relpred(obs, q_truth)
        if fp is not None:
            fp_ba.append(fp)

    # prompt files (one rep: KG is readable)
    os.makedirs(f"{out}/readable/ans/opus", exist_ok=True)
    os.makedirs(f"{out}/readable/ans/qwen", exist_ok=True)
    files = {}
    for seed in SEEDS:
        sp = splits[seed]
        obs, G, typed = obs_cache[seed]
        for k in K_SHOTS:
            prompt, truth, q_order = build_prompt(
                G, typed, sp["support"], sp["query"], k, id2ent, id2rel, rel2id,
                KHOP, MAX_JOINT, MAX_CTX_TRIPLES)
            fn = f"{dataset}/readable/seed{seed}_k{k}.txt"
            with open(f"{OUT_BASE}/{fn}", 'w') as fh:
                fh.write(prompt)
            files[fn] = {
                "dataset": dataset, "rep": "readable", "seed": seed, "k": k,
                "truth": truth, "khop": KHOP, "max_joint": MAX_JOINT,
                "query_triples": [[int(h), int(r), int(t)] for (h, r, t) in q_order],
                "support_triples": [[int(h), int(r), int(t)] for (h, r, t) in sp["support"][:k]],
            }

    n_classes = n_rel
    chance = 1.0 / n_classes
    man = {"dataset": dataset, "task": "relation_prediction",
           "seeds": SEEDS, "k_shots": K_SHOTS, "khop": KHOP, "max_joint": MAX_JOINT,
           "nq": NQ, "n_entities": n_ent, "n_relations": n_rel, "chance": chance,
           "representations": REPS,
           "rel_tokens": {rel_token(rid): id2rel[rid] for rid in sorted(id2rel)},
           "files": files,
           "distmult": {"bal_acc": dm_ba, "hits1": dm_h1, "mrr": dm_mrr,
                        "dim": DM_DIM, "epochs": DM_EPOCHS},
           "freq_prior": {"bal_acc": fp_ba},
           "metric": "balanced_accuracy (primary) + Hits@1/MRR (ranking baselines)"}
    with open(f"{out}/manifest.json", 'w') as fh:
        json.dump(man, fh, indent=0)

    def fmt(v):
        return f"{np.mean(v):.3f}+/-{np.std(v):.3f}" if v else "-"
    print(f"[{dataset}] relation-prediction: entities={n_ent}, relations={n_rel}, "
          f"chance(bal-acc floor)={chance:.3f}, khop={KHOP}, nq={NQ}, seeds={SEEDS}")
    print(f"  DistMult     bal-acc: {fmt(dm_ba)}   Hits@1: {fmt(dm_h1)}   MRR: {fmt(dm_mrr)}")
    print(f"  freq-prior   bal-acc: {fmt(fp_ba)}   (most-common relation)")
    print(f"  wrote {len(files)} prompt files + manifest -> {out}")
    return man


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'UMLS'
    os.makedirs(OUT_BASE, exist_ok=True)
    run(dataset)
    print("run run_qwen.py / run_opus_cli.py on each seed*_k*.txt; "
          "score with score_kg_icl.py")
