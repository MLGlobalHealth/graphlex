"""ULTRA (arXiv 2310.04562) zero-shot specialist FM baseline for the graphlex KG
relation-prediction track on UMLS. Runs on clpc35 (GPU), NO LLM.

Consumes the EXACT same query pairs / leakage-stripped observed graph / seeds as the
graphlex KG arm by reading the manifest written by eval/kg_icl.py:
  - per (seed): query_triples (the [h,r,t] queries) + support_triples are stripped from
    the observed graph by undirected (h,t) pair (same leakage rule as DistMult/LLM arms);
  - relation ids are the sorted-vocab ids from load_kg (deterministic, matches manifest);
  - for each query pair (h,t), ULTRA scores every candidate relation r via its (h,r,t)
    triple score on the leakage-stripped observed graph, argmax_r -> prediction, rank of
    the true r -> Hits@1 / MRR. Balanced accuracy = macro per-relation-class recall.

This builds the ULTRA `Data` object exactly as ULTRA's own transductive loader does
(doubled relations h,t,r + r+num_rel for inverses, build_relation_graph for the
relation-of-relations graph), loads a public checkpoint, and calls model(data, batch).

Smoke validation (mode=smoke): score known UMLS test triples (full graph) for TAIL
prediction and confirm ULTRA ranks them far above random (MRR/Hits@10 >> chance) before
trusting downstream relation-prediction numbers.

Run on clpc35:
  /home/scratch/gpfn_venv/bin/python eval/ultra_kg.py smoke   # sanity check
  /home/scratch/gpfn_venv/bin/python eval/ultra_kg.py run     # relation prediction
"""
import os, sys, json
import numpy as np
import torch

ULTRA_DIR = '/home/scratch/ultra_work/ULTRA'
sys.path.insert(0, ULTRA_DIR)
from torch_geometric.data import Data
from ultra.models import Ultra
from ultra.tasks import build_relation_graph

KG_BASE = '/home/scratch/kg_data'
OUT_BASE = '/home/scratch/bench_out/kg_icl'
CKPT = f'{ULTRA_DIR}/ckpts/ultra_4g.pth'   # 4g: trained on FB15k237,WN18RR,CoDExMedium,NELL995

MODEL_CFG = dict(
    rel_model_cfg=dict(class_='RelNBFNet', input_dim=64,
                       hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult',
                       aggregate_func='sum', short_cut=True, layer_norm=True),
    entity_model_cfg=dict(class_='EntityNBFNet', input_dim=64,
                          hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult',
                          aggregate_func='sum', short_cut=True, layer_norm=True),
)


def load_kg(dataset):
    """Identical vocab/id construction to eval/kg_icl.load_kg (sorted vocab)."""
    def read(split):
        out = []
        path = f"{KG_BASE}/{dataset}/{split}.txt"
        if not os.path.exists(path):
            return out
        for ln in open(path):
            p = ln.rstrip('\n').split('\t')
            if len(p) != 3:
                continue
            out.append(tuple(p))
        return out
    train_s, valid_s, test_s = read('train'), read('valid'), read('test')
    all_s = train_s + valid_s + test_s
    ents = sorted({h for h, r, t in all_s} | {t for h, r, t in all_s})
    rels = sorted({r for h, r, t in all_s})
    ent2id = {e: i for i, e in enumerate(ents)}
    rel2id = {r: i for i, r in enumerate(rels)}
    id2rel = {i: r for r, i in rel2id.items()}
    enc = lambda S: [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in S]
    return (enc(all_s), enc(train_s), enc(test_s), ent2id, rel2id, id2rel)


def build_model(device):
    rc = dict(MODEL_CFG['rel_model_cfg']); rc['class'] = rc.pop('class_')
    ec = dict(MODEL_CFG['entity_model_cfg']); ec['class'] = ec.pop('class_')
    model = Ultra(rel_model_cfg=rc, entity_model_cfg=ec)
    state = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(state['model'])
    return model.to(device).eval()


def make_data(obs_triples, n_ent, n_rel, device):
    """Build the ULTRA inference Data from observed (h,r,t) triples, mirroring ULTRA's
    transductive loader: edges doubled with inverse relations (r and r+n_rel), num_nodes
    fixed at the FULL entity vocab so query entities always exist, relation_graph built."""
    if not obs_triples:
        raise ValueError("empty observed graph")
    tri = torch.tensor(obs_triples, dtype=torch.long)        # (E,3) as (h,r,t)
    h, r, t = tri[:, 0], tri[:, 1], tri[:, 2]
    edge_index = torch.stack([torch.cat([h, t]), torch.cat([t, h])])   # (2, 2E)
    edge_type = torch.cat([r, r + n_rel])                              # (2E,)
    data = Data(edge_index=edge_index, edge_type=edge_type,
                num_nodes=n_ent, num_relations=n_rel * 2)
    data = build_relation_graph(data)
    return data.to(device)


@torch.no_grad()
def score_relations(model, data, h, t, n_rel, device):
    """ULTRA scores for all n_rel candidate relations of the pair (h,t). One batch row
    per candidate (so each candidate's own relation drives its relation representation):
    batch shape (n_rel, 1, 3) of triples (h, t, r). Returns scores over relations."""
    rids = torch.arange(n_rel, device=device)
    triples = torch.stack([torch.full((n_rel,), h, device=device),
                           torch.full((n_rel,), t, device=device),
                           rids], dim=1)            # (n_rel, 3) as (h,t,r)
    batch = triples.unsqueeze(1)                    # (n_rel, 1, 3)
    score = model(data, batch)                      # (n_rel, 1)
    return score.squeeze(1).float().cpu().numpy()   # (n_rel,)


# --------------------------------------------------------------------------- smoke
@torch.no_grad()
def smoke(dataset='UMLS'):
    """Sanity: on the FULL UMLS graph, do TAIL prediction for known test triples and
    confirm ULTRA ranks the true tail far above random (chance MRR ~ 1/n_ent)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (all_tr, train_tr, test_tr, ent2id, rel2id, id2rel) = load_kg(dataset)
    n_ent, n_rel = len(ent2id), len(rel2id)
    model = build_model(device)
    # inference graph = train+valid triples (exclude test, the standard transductive eval)
    obs = [tr for tr in all_tr if tr not in set(map(tuple, test_tr))]
    data = make_data(obs, n_ent, n_rel, device)
    rng = np.random.RandomState(0)
    sample = [test_tr[i] for i in rng.permutation(len(test_tr))[:100]]
    ranks = []
    for (h, r, t) in sample:
        # all_negative tail batch: fix (h,r), vary tail over all entities
        cand = torch.arange(n_ent, device=device)
        triples = torch.stack([torch.full((n_ent,), h, device=device), cand,
                               torch.full((n_ent,), r, device=device)], dim=1)  # (n_ent,3) (h,t,r)
        score = model(data, triples.unsqueeze(0)).squeeze(0).float().cpu().numpy()
        rank = int((score > score[t]).sum()) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    h10 = float(np.mean(ranks <= 10))
    h1 = float(np.mean(ranks <= 1))
    chance_mrr = float(np.mean([1.0 / ((n_ent + 1) / 2) for _ in ranks]))
    print(f"[SMOKE {dataset}] tail-pred on {len(sample)} known test triples "
          f"(full graph, {n_ent} entities):")
    print(f"  MRR={mrr:.3f}  Hits@1={h1:.3f}  Hits@10={h10:.3f}  "
          f"(random MRR ~ {chance_mrr:.4f}, random Hits@10 ~ {10.0/n_ent:.3f})")
    ok = (mrr > 10 * chance_mrr) and (h10 > 5 * (10.0 / n_ent))
    print(f"  SMOKE VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(ULTRA {'ranks known triples far above random' if ok else 'NOT above random — STOP'})")
    return ok, dict(mrr=mrr, hits1=h1, hits10=h10, n=len(sample), n_ent=n_ent,
                    chance_mrr=chance_mrr, ckpt=os.path.basename(CKPT))


# ----------------------------------------------------------------------------- run
@torch.no_grad()
def run(dataset='UMLS'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (all_tr, train_tr, test_tr, ent2id, rel2id, id2rel) = load_kg(dataset)
    n_ent, n_rel = len(ent2id), len(rel2id)
    man = json.load(open(f"{OUT_BASE}/{dataset}/manifest.json"))
    seeds = man['seeds']
    all_set = list(all_tr)

    model = build_model(device)
    per_seed = {}
    for seed in seeds:
        # use the seed's k_max file (queries identical across k; support is the full set)
        fmeta = man['files'][f"{dataset}/readable/seed{seed}_k{max(man['k_shots'])}.txt"]
        q_triples = [tuple(x) for x in fmeta['query_triples']]
        s_triples = [tuple(x) for x in fmeta['support_triples']]
        removed = {frozenset((h, t)) for (h, r, t) in q_triples + s_triples}
        obs = [(h, r, t) for (h, r, t) in all_set if frozenset((h, t)) not in removed]
        data = make_data(obs, n_ent, n_rel, device)
        truths, rankings = [], []
        for (h, r, t) in q_triples:
            scores = score_relations(model, data, h, t, n_rel, device)
            order = np.argsort(-scores).tolist()    # best-first relation ids
            rankings.append(order)
            truths.append(r)
        # metrics
        preds = [rk[0] for rk in rankings]
        by = {}
        for i, ti in enumerate(truths):
            by.setdefault(ti, []).append(i)
        recs = [np.mean([preds[i] == lab for i in ids]) for lab, ids in by.items()]
        ba = float(np.mean(recs))
        h1 = float(np.mean([rankings[i][0] == truths[i] for i in range(len(truths))]))
        mrr = float(np.mean([1.0 / (rankings[i].index(truths[i]) + 1) for i in range(len(truths))]))
        per_seed[seed] = dict(bal_acc=ba, hits1=h1, mrr=mrr, nq=len(truths))
        # also persist per-query predictions for transparency
        per_seed[seed]['preds'] = [int(p) for p in preds]
        per_seed[seed]['truths'] = [int(x) for x in truths]
        print(f"  seed {seed}: bal_acc={ba:.3f}  Hits@1={h1:.3f}  MRR={mrr:.3f}  nq={len(truths)}")

    ba_v = [per_seed[s]['bal_acc'] for s in seeds]
    h1_v = [per_seed[s]['hits1'] for s in seeds]
    mrr_v = [per_seed[s]['mrr'] for s in seeds]
    print(f"\n[ULTRA {os.path.basename(CKPT)} | {dataset} relation prediction]")
    print(f"  bal-acc {np.mean(ba_v):.3f}+/-{np.std(ba_v):.3f}  "
          f"Hits@1 {np.mean(h1_v):.3f}+/-{np.std(h1_v):.3f}  "
          f"MRR {np.mean(mrr_v):.3f}+/-{np.std(mrr_v):.3f}")

    result = dict(checkpoint=os.path.basename(CKPT), bal_acc=ba_v, hits1=h1_v, mrr=mrr_v,
                  seeds=seeds, per_seed={str(s): per_seed[s] for s in seeds},
                  note="zero-shot inductive; same leakage-stripped graph + query pairs as graphlex KG arm")
    # write the ULTRA result block + a standalone json
    outdir = f"{OUT_BASE}/{dataset}"
    with open(f"{outdir}/ultra_result.json", 'w') as fh:
        json.dump(result, fh, indent=2)
    # splice an "ultra" block into the manifest (parallel to "distmult")
    man['ultra'] = result
    with open(f"{outdir}/manifest.json", 'w') as fh:
        json.dump(man, fh, indent=0)
    print(f"  wrote {outdir}/ultra_result.json + spliced 'ultra' block into manifest.json")
    return result


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
    ds = sys.argv[2] if len(sys.argv) > 2 else 'UMLS'
    if mode == 'smoke':
        ok, m = smoke(ds)
        sys.exit(0 if ok else 2)
    elif mode == 'run':
        run(ds)
    else:
        print("usage: ultra_kg.py {smoke|run} [UMLS]")
