"""Score the node-level ICL track: graphlex+LLM (Opus + Qwen) vs trained logreg,
per dataset, BALANCED accuracy, mean +/- std across seeds.

Mirrors score_labelcurve.py but uses _common.bal_acc (balanced accuracy is the
load-bearing metric for the node track — see NODE_TRACK_PLAN.md) and reads the
node_icl manifest (truth + logreg-balanced-acc curves) and ans/<model>/seed*_k*.ans.

The node-track answer files need NO new parser: the LLM emits the SAME '<id> <CLASS>'
lines as every other track, so _common.parse_ans handles them as-is. The only change
vs score_labelcurve.py is swapping raw accuracy for bal_acc (the manifest already
stores logreg as balanced accuracy, computed in node_icl.py).

Run: /home/scratch/fmsn-dev/.venv/bin/python eval/score_node_icl.py [DATASET]
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_ans, bal_acc

B = '/home/scratch/bench_out/node_icl'


def llm_scores(man, model, rep):
    """{k: [balanced acc per seed]} for a model's ans dir, for one representation.
    Per-rep answers live at <B>/<dataset>/<rep>/ans/<model>/seed*_k*.ans."""
    out = {}
    for fn, meta in man['files'].items():
        if meta.get('rep') != rep:
            continue
        # fn = '<dataset>/<rep>/seedX_kY.txt'
        ans = f"{B}/{os.path.dirname(fn)}/ans/{model}/{os.path.basename(fn).replace('.txt', '.ans')}"
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
    lr = man.get('logreg', {}); gc = man.get('gcn', {})
    reps = man.get('representations', ['opaque'])
    print(f"\n=== {dataset} node-ICL (chance {ch:.3f}, {man['n_classes']} classes, "
          f"khop={man['khop']}, BALANCED acc, mean+/-std) ===")
    print(f"  text alignment: {man.get('text_alignment','-')}")
    # logreg + GCN are representation-independent (raw Planetoid features).
    print(f"  -- representation-independent baselines --")
    print(f"  {'K/cls':>5} {'logreg':>14} {'GCN(few-shot)':>14}")
    for k in man['k_shots']:
        print(f"  {k:>5} {_fmt(lr.get(str(k), [])):>14} {_fmt(gc.get(str(k), [])):>14}")
    # LLM arms, split by representation (opaque word-ids vs readable text).
    for rep in reps:
        opus = llm_scores(man, 'opus', rep)
        qwen = llm_scores(man, 'qwen', rep)
        print(f"  -- graphlex+LLM, rep='{rep}' --")
        print(f"  {'K/cls':>5} {'Opus':>14} {'Qwen':>14}")
        for k in man['k_shots']:
            print(f"  {k:>5} {_fmt(opus.get(k, [])):>14} {_fmt(qwen.get(k, [])):>14}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'Cora')
