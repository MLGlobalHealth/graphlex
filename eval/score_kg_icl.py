"""Score the KG relation-prediction ICL track: graphlex+LLM (Opus + Qwen) vs the
DistMult KG-embedding baseline vs the frequency prior, per dataset, BALANCED accuracy
(primary) + Hits@1 / MRR for the ranking baseline, mean +/- std across seeds.

Analogous to score_edge_icl.py. The LLM answer files need NO new parser: the LLM emits
the SAME '<id> <CLASS>' lines (here CLASS = a relation token R00..) as every other
track, so _common.parse_ans handles them and _common.bal_acc scores them. The LLM is a
MULTI-CLASS classifier here (one relation token per query), so it has no ranking; we
report balanced accuracy for the LLM arms and balanced-acc + Hits@1/MRR for DistMult.

Run: /home/scratch/fmsn-dev/.venv/bin/python eval/score_kg_icl.py [DATASET]
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_ans, bal_acc

B = '/home/scratch/bench_out/kg_icl'


def llm_scores(man, model, rep):
    """{k: [balanced acc per seed]} for a model's ans dir, for one representation."""
    out = {}
    for fn, meta in man['files'].items():
        if meta.get('rep') != rep:
            continue
        ans = (f"{B}/{os.path.dirname(fn)}/ans/{model}/"
               f"{os.path.basename(fn).replace('.txt', '.ans')}")
        if not os.path.exists(ans):
            continue
        pred = parse_ans(ans)
        if not pred:
            continue
        acc = bal_acc(meta['truth'], pred)
        if acc is not None:
            out.setdefault(meta['k'], []).append(acc)
    return out


def _fmt(vals):
    return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}" if vals else "         -"


def main(dataset):
    base = f"{B}/{dataset}"
    man = json.load(open(f"{base}/manifest.json"))
    ch = man['chance']
    reps = man.get('representations', ['readable'])
    dm = man.get('distmult', {}); fp = man.get('freq_prior', {})
    print(f"\n=== {dataset} KG-ICL / relation prediction "
          f"(entities={man['n_entities']}, relations={man['n_relations']}, "
          f"bal-acc chance-floor {ch:.3f}, khop={man['khop']}, BALANCED acc, "
          f"mean+/-std) ===")
    # representation-independent baselines
    print(f"  -- representation-independent baselines --")
    print(f"  {'baseline':>18} {'bal-acc':>14} {'Hits@1':>14} {'MRR':>14}")
    print(f"  {'DistMult (KG-emb)':>18} {_fmt(dm.get('bal_acc', [])):>14} "
          f"{_fmt(dm.get('hits1', [])):>14} {_fmt(dm.get('mrr', [])):>14}")
    print(f"  {'freq-prior':>18} {_fmt(fp.get('bal_acc', [])):>14} "
          f"{'(n/a)':>14} {'(n/a)':>14}")
    print(f"  {'ULTRA (zero-shot)':>18} {'ENV-PENDING':>14} {'ENV-PENDING':>14} "
          f"{'ENV-PENDING':>14}   <- see KG_TRACK_PLAN.md")
    # LLM arms (balanced accuracy), per rep
    for rep in reps:
        opus = llm_scores(man, 'opus', rep)
        qwen = llm_scores(man, 'qwen', rep)
        print(f"  -- graphlex+LLM, rep='{rep}' (balanced acc) --")
        print(f"  {'K':>5} {'Opus':>14} {'Qwen':>14}")
        for k in man['k_shots']:
            print(f"  {k:>5} {_fmt(opus.get(k, [])):>14} {_fmt(qwen.get(k, [])):>14}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'UMLS')
