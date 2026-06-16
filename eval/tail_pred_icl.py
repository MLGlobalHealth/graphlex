"""Knowledge-graph TAIL-PREDICTION few-shot ICL track: graphlex+LLM vs a small
KG-embedding baseline (DistMult, numpy) vs a frequency/prior baseline, on a small
multi-relational KG (UMLS). This is ULTRA's NATIVE task -- entity ranking given a
relation -- so it is the HOME-TURF comparison for the specialist FM foil.

Completes the KG 2x2: {relation-pred (kg_icl.py), tail-pred (here)} x {graphlex+LLM
vs ULTRA}. relation-pred is OFF ULTRA's objective (rank 46 relations for a fixed
entity pair); tail-pred IS ULTRA's objective (rank the 135 candidate tails for a
fixed (h, r)). graphlex+LLM spans BOTH cells of the same results table while the
specialist FM is pinned to its home task.

Sibling of kg_icl.py -- read that first. SAME KG loading (load_kg, sorted vocab),
split/seed protocol, leakage-stripping by undirected (h,t) pair, graphlex
verbalization of the joint neighborhood + appended typed-triples context, driver-
compatible seed*_k*.txt prompt format, and manifest. The DIFFERENCES:

  * a "query" is an ordered pair (HEAD h, RELATION r); the LABEL is the TAIL entity t.
    The model must RANK candidate tail entities and put the true t first.
  * few-shot support = K labeled (h, r -> t) examples, each verbalized like a query.
  * VERBALIZATION (the graphlex angle): for the head h we verbalize the UNTYPED
    structural skeleton of h's 1-hop neighborhood via graphlex facts()/
    verbalize(focus='structure'), then APPEND a readable TYPED-TRIPLES context line
    listing the observed triples incident to h (the query triple (h, r, t) -- by
    undirected (h,t) pair -- is NEVER shown). Same appending pattern as kg_icl.py;
    graphlex core is NOT modified.
  * the prompt presents the CANDIDATE ENTITY MENU (UMLS = 135 entities, mapped to
    tokens E000..E134 with readable names) and asks for a RANKED list of the top
    entities per query. Output: one line per query, '<id> <E007,E003,E120,...>'.
    A NEW ranked-output parser (score_tail_pred.parse_ranked) extracts these.

METRICS (rank-based, standard KG link prediction): Hits@1, Hits@10, MRR of the true
tail. Primary = MRR + Hits@1. Mean over seeds.

BASELINES (run, no LLM):
  1. DistMult (numpy, tiny epoch budget) -- score(h,r,t)=sum(e_h*w_r*e_t); rank all
     candidate tails for (h,r). Trained on observed (leakage-stripped) triples only.
     (Same training routine as kg_icl.py; here scored as TAIL ranking.)
  2. freq-prior -- rank tails by how often they appear as the tail of relation r in
     the observed graph (a strong, standard KG baseline). Ties / unseen tails fall
     back to global tail frequency.
ULTRA (the FM foil, NATIVE task) is ENV-PENDING -- see TAIL_PRED_PLAN.md for the
matched-run spec (same query (h,r) pairs / seeds / leakage-stripped graph; ULTRA
ranks all entities for (h,r,?)). ultra_kg.py already has the tail-ranking machinery
(its smoke()); the matched run is a `tail` mode hook over this manifest.

Run:  /home/scratch/fmsn-dev/.venv/bin/python eval/tail_pred_icl.py [DATASET]
      DATASET in {UMLS, Nations, Kinship}; default UMLS. Env SMOKE=1 -> tiny grid.
      Calls NO LLM.
"""
import os, sys, json
import numpy as np
import networkx as nx

sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
from graphlex import facts, verbalize

# reuse the SIBLING track's KG loading + DistMult training verbatim (do not reinvent).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg_icl import (load_kg, observed_triples, build_nx, joint_nx, train_distmult,
                    DM_DIM, DM_EPOCHS, DM_LR, DM_REG, DM_NEG)

OUT_BASE = '/home/scratch/bench_out/tail_pred_icl'
SEEDS = [11, 22, 33]            # >=3 seeds (full)
K_SHOTS = [1, 3]               # labeled (h, r -> t) example pairs per query block
KHOP = 1                        # head-neighborhood radius
MAX_JOINT = 16                  # cap head-neighborhood node count
MAX_CTX_TRIPLES = 24            # cap typed-context triples shown per head
NQ = 20                         # query (h,r) pairs (sampled from the test split)
TOPK_ASK = 10                   # ask the LLM for its top-10 ranked tails
REPS = ['readable']             # KG entities ARE human-readable -> one rep

if os.environ.get('SMOKE'):
    SEEDS, K_SHOTS, NQ = [11, 22], [1, 3], 12
if os.environ.get('FULL'):
    SEEDS, K_SHOTS, NQ = [11, 22, 33, 44, 55], [1, 3, 5], 40


# entity CLASS tokens: E000..E134 (zero-padded to the vocab width).
def ent_token(eid, n_ent):
    w = len(str(n_ent - 1))
    return f"E{eid:0{w}d}"


# --- typed-triples context line for the HEAD (appended; graphlex core untouched) ---
def head_typed_context(typed, h, id2ent, id2rel, exclude_pair, max_triples):
    """Readable list of observed typed triples incident to the head h (the query
    pair's own triples -- by undirected (h,t) -- excluded). Parallel to kg_icl's
    typed_context but anchored on the single head entity."""
    seen = set(); lines = []
    for (hh, rr, tt) in typed.get(h, []):
        if frozenset((hh, tt)) == exclude_pair:
            continue
        key = (hh, rr, tt)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"({id2ent[hh]}) -[{id2rel[rr]}]-> ({id2ent[tt]})")
        if len(lines) >= max_triples:
            break
    if not lines:
        return "Known typed triples around the head (excluding the query): (none observed)"
    return ("Known typed triples around the head (excluding the query):\n    "
            + "\n    ".join(lines))


def verbalize_query(G, typed, h, r, t, id2ent, id2rel, khop, max_joint, max_ctx):
    """Verbalize a tail-prediction query/example: untyped structural skeleton of the
    head's neighborhood (graphlex) + the readable typed-triples context (appended) +
    the stated relation r. The TAIL is never shown (it is the answer). #0 = head.
    `t` is the (h,t) pair to exclude from leakage (the true tail for an example; for a
    query it is the held-out answer, excluded identically)."""
    # head-only neighborhood: reuse joint_nx with h==t so it builds the k-hop ball of h.
    H, glob = joint_nx(G, h, h, khop, max_joint)
    struct = verbalize(facts(H), focus='structure')
    ctx = head_typed_context(typed, h, id2ent, id2rel, frozenset((h, t)), max_ctx)
    return (f"HEAD = #0 = '{id2ent[h]}'  ,  RELATION = '{id2rel[r]}'  "
            f"(predict the TAIL entity t such that HEAD -[{id2rel[r]}]-> t).\n"
            f"Untyped {khop}-hop neighborhood skeleton of the head: {struct}\n"
            f"{ctx}")


# --- tail-ranking metrics (rank-based) ----------------------------------------
def tail_ranking_metrics(rankings, truths, filt_sets=None):
    """Hits@1 / Hits@10 / MRR given per-query tail rankings (best-first entity ids)
    and the true tail id per query. If filt_sets is given (per-query set of OTHER
    true tails for (h,r) to filter out before ranking -- the standard 'filtered'
    setting), they are removed from the ranking before computing the rank of truth."""
    ranks = []
    for i, t in enumerate(truths):
        rk = rankings[i]
        if filt_sets is not None:
            others = filt_sets[i]
            rk = [e for e in rk if e == t or e not in others]
        ranks.append(rk.index(t) + 1)
    ranks = np.array(ranks)
    h1 = float(np.mean(ranks <= 1))
    h10 = float(np.mean(ranks <= 10))
    mrr = float(np.mean(1.0 / ranks))
    return h1, h10, mrr


def distmult_tailrank(E, W, query_hr):
    """For each (h,r) query, rank ALL candidate tails by DistMult score
    score(h,r,e) = <E_h * W_r, E_e> over every entity e. Returns best-first entity-id
    rankings per query."""
    rankings = []
    for (h, r) in query_hr:
        scores = (E * (E[h] * W[r])[None, :]).sum(1)   # (n_ent,) score over all tails
        order = np.argsort(-scores)
        rankings.append(order.tolist())
    return rankings


def freq_prior_tailrank(obs_triples, n_ent, query_hr):
    """Rank candidate tails by how often they appear as the tail of relation r in the
    observed graph (standard KG freq baseline). Within-r ties and unseen tails are
    broken by GLOBAL tail frequency, then entity id. Returns best-first rankings."""
    # per-relation tail counts + global tail counts
    rel_tail = {}                      # r -> np.array(n_ent) tail counts under r
    glob_tail = np.zeros(n_ent)
    for (h, r, t) in obs_triples:
        rel_tail.setdefault(r, np.zeros(n_ent))[t] += 1
        glob_tail[t] += 1
    # stable composite key: primary per-r count, secondary global count, tertiary -id
    rankings = []
    for (h, r) in query_hr:
        rt = rel_tail.get(r, np.zeros(n_ent))
        # sort by (-rt, -glob, id) -> lexsort with last key primary
        order = np.lexsort((np.arange(n_ent), -glob_tail, -rt))
        rankings.append(order.tolist())
    return rankings


# --- splits (same de-dup-by-(h,t) protocol as kg_icl.make_splits) -------------
def make_splits(test_triples, all_triples, seeds, k_max, nq):
    """Per seed: nq query triples (from test) + k_max support triples (from the rest),
    distinct undirected (h,t) pairs. Returns {seed: {support, query}} of (h,r,t)."""
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
def build_prompt(G, typed, support, query, k, id2ent, id2rel, n_ent,
                 khop, max_joint, max_ctx, topk_ask):
    """One ICL tail-prediction prompt for a (seed,k). Returns (prompt, truth_list,
    query_order). truth_list = [[qid, TAIL_TOKEN], ...]."""
    # candidate ENTITY MENU (full vocab): token -> readable name
    ent_ids = sorted(id2ent.keys())
    menu = "\n".join(f"  {ent_token(eid, n_ent)} = {id2ent[eid]}" for eid in ent_ids)
    TASK = (
        "Each item is an ordered pair (HEAD = #0, RELATION) in a knowledge graph. "
        "Your job is TAIL PREDICTION: rank which TAIL entity t completes the triple "
        "HEAD -[RELATION]-> t, choosing from the candidate entities below. You are "
        "shown the untyped structural skeleton of the HEAD's local neighborhood plus "
        "the known TYPED triples incident to the HEAD (the query triple itself is "
        "never shown). Learn the pattern from the labeled examples, then rank tails "
        "for each query.\n\n"
        "CANDIDATE ENTITIES (answer with the tokens on the left):\n"
        f"{menu}\n\n"
        f"OUTPUT FORMAT: one line per query, exactly '<id> <t1,t2,...,t{topk_ask}>' "
        f"where t1..t{topk_ask} are your TOP-{topk_ask} ranked entity tokens "
        "(best first), comma-separated, no spaces, e.g. '0 E007,E003,E120'. Put your "
        "single best guess FIRST. No other text.")
    L = [TASK, "", "=== LABELED EXAMPLES ==="]
    for (h, r, t) in support[:k]:
        body = verbalize_query(G, typed, h, r, t, id2ent, id2rel, khop, max_joint, max_ctx)
        L.append(f"[answer: {ent_token(t, n_ent)} = {id2ent[t]}]\n{body}\n")
    L.append("=== QUERIES (rank tails for each) ===")
    truth = []
    order = np.random.RandomState(12345).permutation(len(query))
    q_order = []
    for qi, oi in enumerate(order):
        h, r, t = query[oi]
        body = verbalize_query(G, typed, h, r, t, id2ent, id2rel, khop, max_joint, max_ctx)
        L.append(f"Query {qi}:\n{body}\n")
        truth.append([qi, ent_token(t, n_ent)])
        q_order.append((h, r, t))
    return "\n".join(L), truth, q_order


def run(dataset):
    out = f"{OUT_BASE}/{dataset}"
    os.makedirs(out, exist_ok=True)
    (all_tr, train_tr, test_tr, ent2id, rel2id, id2rel, id2ent) = load_kg(dataset)
    n_ent, n_rel = len(ent2id), len(rel2id)

    splits = make_splits(test_tr, all_tr, SEEDS, max(K_SHOTS), NQ)

    # filtered-setting helper: ALL true tails per (h,r) across the full graph, so other
    # correct tails don't penalise the rank of the held-out truth (standard KG eval).
    hr_tails = {}
    for (h, r, t) in all_tr:
        hr_tails.setdefault((h, r), set()).add(t)

    # baselines per seed: observed graph removes that seed's query + support pairs.
    dm_h1 = []; dm_h10 = []; dm_mrr = []
    fp_h1 = []; fp_h10 = []; fp_mrr = []
    obs_cache = {}
    for seed in SEEDS:
        sp = splits[seed]
        removed = {frozenset((h, t)) for (h, r, t) in sp["query"] + sp["support"]}
        obs = observed_triples(all_tr, removed)
        G, typed = build_nx(obs)
        obs_cache[seed] = (obs, G, typed)
        q_hr = [(h, r) for (h, r, t) in sp["query"]]
        q_truth = [t for (h, r, t) in sp["query"]]
        # filtered other-tails per query (exclude the held-out truth itself)
        filt = [hr_tails.get((h, r), set()) - {t} for (h, r, t) in sp["query"]]
        # DistMult (trained on observed triples only)
        E, W = train_distmult(obs, n_ent, n_rel, seed)
        dm_rank = distmult_tailrank(E, W, q_hr)
        h1, h10, mrr = tail_ranking_metrics(dm_rank, q_truth, filt)
        dm_h1.append(h1); dm_h10.append(h10); dm_mrr.append(mrr)
        # frequency prior
        fp_rank = freq_prior_tailrank(obs, n_ent, q_hr)
        h1, h10, mrr = tail_ranking_metrics(fp_rank, q_truth, filt)
        fp_h1.append(h1); fp_h10.append(h10); fp_mrr.append(mrr)

    # prompt files (one rep: KG is readable)
    os.makedirs(f"{out}/readable/ans/opus", exist_ok=True)
    os.makedirs(f"{out}/readable/ans/qwen", exist_ok=True)
    files = {}
    for seed in SEEDS:
        sp = splits[seed]
        obs, G, typed = obs_cache[seed]
        for k in K_SHOTS:
            prompt, truth, q_order = build_prompt(
                G, typed, sp["support"], sp["query"], k, id2ent, id2rel, n_ent,
                KHOP, MAX_JOINT, MAX_CTX_TRIPLES, TOPK_ASK)
            fn = f"{dataset}/readable/seed{seed}_k{k}.txt"
            with open(f"{OUT_BASE}/{fn}", 'w') as fh:
                fh.write(prompt)
            # filtered other-tails per query (for the scorer), in query order
            filt = [sorted(hr_tails.get((h, r), set()) - {t})
                    for (h, r, t) in q_order]
            files[fn] = {
                "dataset": dataset, "rep": "readable", "seed": seed, "k": k,
                "truth": truth, "khop": KHOP, "max_joint": MAX_JOINT,
                "topk_ask": TOPK_ASK,
                "query_triples": [[int(h), int(r), int(t)] for (h, r, t) in q_order],
                "support_triples": [[int(h), int(r), int(t)] for (h, r, t) in sp["support"][:k]],
                "filter_tails": [[int(e) for e in fs] for fs in filt],
            }

    chance_mrr = float(np.mean([1.0 / ((n_ent + 1) / 2)]))   # ~ rank of random tail
    man = {"dataset": dataset, "task": "tail_prediction",
           "seeds": SEEDS, "k_shots": K_SHOTS, "khop": KHOP, "max_joint": MAX_JOINT,
           "nq": NQ, "n_entities": n_ent, "n_relations": n_rel,
           "topk_ask": TOPK_ASK,
           "chance_mrr": chance_mrr, "chance_hits10": 10.0 / n_ent,
           "representations": REPS,
           "ent_tokens": {ent_token(eid, n_ent): id2ent[eid] for eid in sorted(id2ent)},
           "files": files,
           "distmult": {"hits1": dm_h1, "hits10": dm_h10, "mrr": dm_mrr,
                        "dim": DM_DIM, "epochs": DM_EPOCHS},
           "freq_prior": {"hits1": fp_h1, "hits10": fp_h10, "mrr": fp_mrr},
           "metric": "Hits@1 / Hits@10 / MRR of the true tail (filtered); primary MRR+Hits@1"}
    with open(f"{out}/manifest.json", 'w') as fh:
        json.dump(man, fh, indent=0)

    def fmt(v):
        return f"{np.mean(v):.3f}+/-{np.std(v):.3f}" if v else "-"
    print(f"[{dataset}] tail-prediction: entities={n_ent}, relations={n_rel}, "
          f"chance MRR~{chance_mrr:.4f}, chance Hits@10~{10.0/n_ent:.3f}, "
          f"khop={KHOP}, nq={NQ}, seeds={SEEDS} (filtered ranking)")
    print(f"  DistMult     Hits@1: {fmt(dm_h1)}   Hits@10: {fmt(dm_h10)}   MRR: {fmt(dm_mrr)}")
    print(f"  freq-prior   Hits@1: {fmt(fp_h1)}   Hits@10: {fmt(fp_h10)}   MRR: {fmt(fp_mrr)}")
    print(f"  wrote {len(files)} prompt files + manifest -> {out}")
    return man


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'UMLS'
    os.makedirs(OUT_BASE, exist_ok=True)
    run(dataset)
    print("run run_qwen.py / run_opus_cli.py on each seed*_k*.txt; "
          "score with score_tail_pred.py")
