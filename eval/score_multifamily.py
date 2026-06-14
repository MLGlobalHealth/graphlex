"""Cross-FAMILY label-efficiency ladder: every model that has an ans/<model>/ dir
under labelcurve, scored per domain x k (mean +/- std over seeds), vs logreg.
Auto-discovers model dirs. Uses the unified tolerant parser from _common."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_ans, raw_acc

B = '/home/scratch/bench_out/labelcurve'
man = json.load(open(f"{B}/manifest.json"))
DOMAINS = ['family', 'proteins', 'imdb']
KS = man['k_llm']

# discover model dirs
models = set()
for t in DOMAINS:
    d = f"{B}/{t}/ans"
    if os.path.isdir(d):
        models |= {m for m in os.listdir(d) if os.path.isdir(f"{d}/{m}")}
# order: classical-ref via logreg, then models with a friendly order
order = [m for m in ['opus', 'qwen32', 'qwen', 'gemma2_27b', 'gemma2_9b', 'llama31_8b',
                     'mistral_7b'] if m in models] + sorted(models - {
    'opus', 'qwen32', 'qwen', 'gemma2_27b', 'gemma2_9b', 'llama31_8b', 'mistral_7b'})


def scores(t, model):
    out = {}
    for fn, meta in man['tasks'][t]['files'].items():
        a = f"{B}/{t}/ans/{model}/{os.path.basename(fn).replace('.txt', '.ans')}"
        if os.path.exists(a):
            p = parse_ans(a)
            if p:
                out.setdefault(meta['k'], []).append(raw_acc(meta['truth'], p))
    return out


for t in DOMAINS:
    lr = man['tasks'][t]['logreg']
    ch = man['tasks'][t]['chance']
    print(f"\n=== {t.upper()} (chance {ch:.3f}) — accuracy mean±std, vs logreg ===")
    hdr = f"  {'model':12}" + "".join(f"{('k='+str(k)):>14}" for k in KS)
    print(hdr)
    print(f"  {'logreg':12}" + "".join(
        f"{(f'{np.mean(lr[str(k)]):.3f}' if str(k) in lr and lr[str(k)] else '-'):>14}" for k in KS))
    for m in order:
        s = scores(t, m)
        cells = []
        for k in KS:
            if k in s and s[k]:
                cells.append(f"{np.mean(s[k]):.3f}±{np.std(s[k]):.2f}({len(s[k])})")
            else:
                cells.append("-")
        print(f"  {m:12}" + "".join(f"{c:>14}" for c in cells))

# capability-ladder summary at k=1 on family: who beats logreg?
print("\n=== capability ladder @ family, k=1 (beats logreg?) ===")
lr1 = np.mean(man['tasks']['family']['logreg']['1'])
print(f"  logreg       {lr1:.3f}")
fam = [(m, np.mean(v)) for m in order for v in [scores('family', m).get(1, [])] if v]
for m, acc in sorted(fam, key=lambda x: -x[1]):
    print(f"  {m:12} {acc:.3f}  ({'BEATS' if acc > lr1 else 'below'} logreg)")

json.dump({t: {m: {str(k): scores(t, m).get(k, []) for k in KS} for m in order} for t in DOMAINS},
          open(f"{B}/multifamily_scores.json", 'w'), indent=0)
print(f"\n-> {B}/multifamily_scores.json ; models found: {order}")
