"""Score the edge / link-prediction ICL track: graphlex+LLM (Opus + Qwen) vs
heuristic-logreg vs trained-GNN-link-predictor, per dataset, BALANCED accuracy AND
AUC, mean +/- std across seeds.

Analogous to score_node_icl.py. The LLM answer files need NO new parser: the LLM
emits the SAME '<id> <CLASS>' lines (here CLASS1 = LINK, CLASS0 = NOLINK) as every
other track, so _common.parse_ans handles them as-is and _common.bal_acc scores them.

AUC for the LLM arms: the LLM emits a hard LINK/NOLINK label (no probability), so its
AUC is the balanced-accuracy-equivalent of a hard binary classifier — we report AUC
only for the probabilistic baselines (heuristic-logreg, GNN) and balanced accuracy
for all arms. (To get a real LLM AUC one would prompt for a confidence; out of scope.)

Run: /home/scratch/fmsn-dev/.venv/bin/python eval/score_edge_icl.py [DATASET]
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_ans, bal_acc

B = '/home/scratch/bench_out/edge_icl'


def llm_scores(man, model, rep):
    """{k: [balanced acc per seed]} for a model's ans dir, for one representation.
    Per-rep answers live at <B>/<dataset>/<rep>/ans/<model>/seed*_k*.ans."""
    out = {}
    for fn, meta in man['files'].items():
        if meta.get('rep') != rep:
            continue
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
    reps = man.get('representations', ['opaque'])
    hl = man.get('heuristic_logreg', {}); gn = man.get('gnn_link', {})
    hl_ba, hl_auc = hl.get('bal_acc', {}), hl.get('auc', {})
    gn_ba, gn_auc = gn.get('bal_acc', {}), gn.get('auc', {})
    print(f"\n=== {dataset} edge-ICL / link prediction (chance {ch:.3f}, "
          f"khop={man['khop']}, BALANCED acc + AUC, mean+/-std) ===")
    print(f"  text alignment: {man.get('text_alignment','-')}")
    # representation-independent baselines (computed on the observed graph)
    print(f"  -- representation-independent baselines (bal-acc / AUC) --")
    print(f"  {'K/cls':>5} {'heur-logreg BA':>15} {'heur-logreg AUC':>16} "
          f"{'GNN-link BA':>13} {'GNN-link AUC':>14}")
    for k in man['k_shots']:
        sk = str(k)
        print(f"  {k:>5} {_fmt(hl_ba.get(sk, [])):>15} {_fmt(hl_auc.get(sk, [])):>16} "
              f"{_fmt(gn_ba.get(sk, [])):>13} {_fmt(gn_auc.get(sk, [])):>14}")
    # LLM arms (balanced accuracy; hard labels -> no probabilistic AUC), per rep.
    for rep in reps:
        opus = llm_scores(man, 'opus', rep)
        qwen = llm_scores(man, 'qwen', rep)
        print(f"  -- graphlex+LLM, rep='{rep}' (balanced acc) --")
        print(f"  {'K/cls':>5} {'Opus':>14} {'Qwen':>14}")
        for k in man['k_shots']:
            print(f"  {k:>5} {_fmt(opus.get(k, [])):>14} {_fmt(qwen.get(k, [])):>14}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'Cora')
