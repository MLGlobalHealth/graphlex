"""Score the fusion-KG killer-app smoke: balanced accuracy (+ AUC where available) per
CONDITION x MODEL, plus the non-LLM structural baseline, with the DECISIVE READ
(fusion vs max(knowledge, anon)) and the memorization gauge (knowledge-only level).

The LLM answer files need no new parser: the same '<id> <CLASS>' lines as every other
graphlex track, so _common.parse_ans + _common.bal_acc handle them. LLM arms have no
ranking -> balanced accuracy only; the structural-logreg baseline reports bal-acc+AUC.

Run: /home/scratch/fmsn-dev/.venv/bin/python eval/score_fusion_kg.py [hetionet]
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_ans, bal_acc

B = '/home/scratch/bench_out/fusion_kg'


def llm_scores(man, model, cond):
    """{k: [balanced acc per seed]} for one model's ans dir under one condition."""
    out = {}
    for fn, meta in man['files'].items():
        if meta.get('condition') != cond:
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


def pooled(scores):
    """Pool all per-(seed,k) bal-accs into one mean+/-std + the raw list."""
    vals = [v for k in scores for v in scores[k]]
    return vals


def _m(vals):
    return float(np.mean(vals)) if vals else None


def _fmt(vals):
    return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f} (n={len(vals)})" if vals else "         -"


def main(dataset='hetionet'):
    base = f"{B}/{dataset}"
    man = json.load(open(f"{base}/manifest.json"))
    conds = man['conditions']
    sl = man.get('structural_logreg', {})
    sl_ba = [v for k in sl.get('bal_acc', {}) for v in sl['bal_acc'][k]]
    sl_auc = [v for k in sl.get('auc', {}) for v in sl['auc'][k]]

    print(f"\n=== {dataset} FUSION-KG killer-app smoke "
          f"(Compound-treats-Disease, {man['n_edges']} edges, "
          f"{man['n_compounds']} compounds / {man['n_diseases']} diseases; "
          f"chance {man['chance']:.3f}, BALANCED acc, pooled over seeds x k) ===")

    print(f"\n  -- non-LLM structural reference (leakage-stripped graph) --")
    print(f"  {'structural-logreg':>20}  bal-acc {_fmt(sl_ba):>22}  AUC {_fmt(sl_auc):>22}")

    rows = {}   # (model, cond) -> pooled list
    for model in ('opus', 'qwen'):
        print(f"\n  -- graphlex+LLM: {model} (balanced acc, per condition) --")
        print(f"  {'condition':>12} {'K=1':>22} {'K=3':>22} {'POOLED':>22}")
        for cond in conds:
            sc = llm_scores(man, model, cond)
            rows[(model, cond)] = pooled(sc)
            k1 = sc.get(1, []); k3 = sc.get(3, [])
            print(f"  {cond:>12} {_fmt(k1):>22} {_fmt(k3):>22} {_fmt(pooled(sc)):>22}")

    # --- THE DECISIVE READ ---
    print(f"\n  === DECISIVE READ ===")
    for model in ('opus', 'qwen'):
        kn = _m(rows.get((model, 'knowledge'), []))
        an = _m(rows.get((model, 'anon'), []))
        fu = _m(rows.get((model, 'fusion'), []))
        if fu is None or (kn is None and an is None):
            print(f"  [{model}] insufficient answers scored (need fusion + >=1 single-signal arm).")
            continue
        singles = [x for x in (kn, an) if x is not None]
        best_single = max(singles)
        gap = fu - best_single
        verdict = "FUSION WINS" if gap > 0 else "fusion does NOT clear both"
        print(f"  [{model}] knowledge-only={fmtn(kn)}  anon-structure={fmtn(an)}  "
              f"fusion={fmtn(fu)}")
        print(f"           fusion - max(knowledge,anon) = {gap:+.3f}  ->  {verdict}")
        # memorization gauge
        if kn is not None:
            mem = ("NEAR-CEILING (task likely memorized -> temporal holdout needed)"
                   if kn >= 0.85 else
                   "moderate" if kn >= 0.65 else "low (knowledge alone weak)")
            print(f"           memorization gauge (knowledge-only abs level): {kn:.3f} -> {mem}")
    print(f"\n  NOTE: smoke uses a RANDOM holdout. A temporal holdout (drugs approved after a "
          f"cutoff) is the rigorous anti-memorization follow-on (see FUSION_SMOKE_PLAN.md).")


def fmtn(x):
    return f"{x:.3f}" if x is not None else "  -  "


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'hetionet')
