"""Score the KG TAIL-PREDICTION ICL track: graphlex+LLM (Opus + Qwen) vs the DistMult
KG-embedding baseline vs the frequency prior vs ULTRA (the FM foil, NATIVE task),
per dataset, with RANK-BASED metrics Hits@1 / Hits@10 / MRR (mean +/- std over seeds).

Sibling of score_kg_icl.py. UNLIKE the relation-prediction scorer, the LLM here emits
a RANKED list of entity tokens per query ('<id> <E007,E003,E120,...>'), so a NEW parser
(parse_ranked, extending the _common.ANS_LINE style) extracts the per-query ranking.
The true tail's rank in that list (filtered: other true (h,r) tails removed) gives
Hits@1 / Hits@10 / MRR. A truth not present in the (truncated top-K) LLM list is
treated as rank infinity (MRR 0, no hit) -- the honest treatment of a top-K answer.

Run: /home/scratch/fmsn-dev/.venv/bin/python eval/score_tail_pred.py [DATASET]
"""
import os, re, sys, json
import numpy as np

B = '/home/scratch/bench_out/tail_pred_icl'

# tolerant ranked-answer line: "0 E007,E003", "Query 0 E007, E003", "0: E007 E003",
# "0) E007,E003" -- id, then a list of entity tokens separated by commas/spaces.
RANK_LINE = re.compile(r'^\s*(?:query\s*)?(\d+)\s*[:.\)\-]?\s+(.*\bE\d+.*)$', re.I)
TOKEN = re.compile(r'E\d+', re.I)


def parse_ranked(path):
    """Parse a ranked-answer file -> {int query id: [UPPER entity tokens, best-first]}.
    Tolerant of 'Query N', punctuation, comma/space separators, and surrounding chatter
    on the line. Order is preserved; duplicate tokens are dropped keeping first."""
    d = {}
    with open(path) as fh:
        for ln in fh:
            m = RANK_LINE.match(ln.strip())
            if not m:
                continue
            qid = int(m.group(1))
            toks = []
            seen = set()
            for tk in TOKEN.findall(m.group(2)):
                u = tk.upper()
                if u not in seen:
                    seen.add(u); toks.append(u)
            if toks:
                d[qid] = toks
    return d


def llm_rank_scores(man, model, rep):
    """{k: {'hits1':[...],'hits10':[...],'mrr':[...]}} over seeds for a model's ans dir.
    Filtered: per query, other true (h,r) tails (manifest 'filter_tails', as tokens via
    ent_tokens) are removed from the LLM ranking before locating the true tail. A truth
    absent from the (top-K) LLM list -> rank infinity (MRR 0, no hit)."""
    n_ent = man['n_entities']
    w = len(str(n_ent - 1))
    tok = lambda eid: f"E{eid:0{w}d}"
    out = {}
    for fn, meta in man['files'].items():
        if meta.get('rep') != rep:
            continue
        ans = (f"{B}/{os.path.dirname(fn)}/ans/{model}/"
               f"{os.path.basename(fn).replace('.txt', '.ans')}")
        if not os.path.exists(ans):
            continue
        pred = parse_ranked(ans)
        if not pred:
            continue
        truth = {int(qid): str(tt).upper() for qid, tt in meta['truth']}
        filt = {i: {tok(e).upper() for e in fl}
                for i, fl in enumerate(meta.get('filter_tails', []))}
        ranks = []
        for qid, tt in truth.items():
            ranked = pred.get(qid, [])
            others = filt.get(qid, set())
            ranked = [e for e in ranked if e == tt or e not in others]
            if tt in ranked:
                ranks.append(ranked.index(tt) + 1)
            else:
                ranks.append(np.inf)        # not in the model's top-K list
        ranks = np.array(ranks, dtype=float)
        d = out.setdefault(meta['k'], {'hits1': [], 'hits10': [], 'mrr': []})
        d['hits1'].append(float(np.mean(ranks <= 1)))
        d['hits10'].append(float(np.mean(ranks <= 10)))
        d['mrr'].append(float(np.mean(1.0 / ranks)))
    return out


def _fmt(vals):
    return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}" if len(vals) else "         -"


def main(dataset):
    base = f"{B}/{dataset}"
    man = json.load(open(f"{base}/manifest.json"))
    reps = man.get('representations', ['readable'])
    dm = man.get('distmult', {}); fp = man.get('freq_prior', {})
    ul = man.get('ultra', {})
    cm, ch10 = man.get('chance_mrr', 0), man.get('chance_hits10', 0)
    print(f"\n=== {dataset} TAIL-prediction (ULTRA's native task) "
          f"(entities={man['n_entities']}, relations={man['n_relations']}, "
          f"chance MRR~{cm:.4f} / Hits@10~{ch10:.3f}, khop={man['khop']}, "
          f"filtered ranking, mean+/-std) ===")
    print(f"  -- representation-independent baselines --")
    print(f"  {'baseline':>20} {'Hits@1':>14} {'Hits@10':>14} {'MRR':>14}")
    print(f"  {'DistMult (KG-emb)':>20} {_fmt(dm.get('hits1', [])):>14} "
          f"{_fmt(dm.get('hits10', [])):>14} {_fmt(dm.get('mrr', [])):>14}")
    print(f"  {'freq-prior':>20} {_fmt(fp.get('hits1', [])):>14} "
          f"{_fmt(fp.get('hits10', [])):>14} {_fmt(fp.get('mrr', [])):>14}")
    if ul:
        ck = ul.get('checkpoint', 'ultra').replace('.pth', '')
        print(f"  {('ULTRA ('+ck+')'):>20} {_fmt(ul.get('hits1', [])):>14} "
              f"{_fmt(ul.get('hits10', [])):>14} {_fmt(ul.get('mrr', [])):>14}"
              f"   <- zero-shot, NATIVE task, matched queries/seeds")
    else:
        print(f"  {'ULTRA (native)':>20} {'ENV-PENDING':>14} {'ENV-PENDING':>14} "
              f"{'ENV-PENDING':>14}   <- see TAIL_PRED_PLAN.md")
    for rep in reps:
        for model, label in (('opus', 'Opus'), ('qwen', 'Qwen')):
            sc = llm_rank_scores(man, model, rep)
            if not sc:
                continue
            print(f"  -- graphlex+{label}, rep='{rep}' (ranked top-{man.get('topk_ask',10)}) --")
            print(f"  {'K':>5} {'Hits@1':>14} {'Hits@10':>14} {'MRR':>14}")
            for k in man['k_shots']:
                d = sc.get(k, {})
                print(f"  {k:>5} {_fmt(d.get('hits1', [])):>14} "
                      f"{_fmt(d.get('hits10', [])):>14} {_fmt(d.get('mrr', [])):>14}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'UMLS')
