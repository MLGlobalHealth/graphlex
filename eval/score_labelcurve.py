"""Score the hardened label-efficiency crossover: graphlex+LLM (Opus + Qwen) vs
logreg, per domain, with mean +/- std across seeds. Reads the manifest for truth +
logreg curves; reads ans/<model>/seed*_k*.ans for each model."""
import os, re, json, glob
import numpy as np

B = '/home/scratch/bench_out/labelcurve'
man = json.load(open(f"{B}/manifest.json"))
LINE = re.compile(r'^\s*(\d+)\s+([A-Za-z0-9_]+)\s*$')


def parse(p):
    d = {}
    for ln in open(p):
        m = LINE.match(ln.strip())
        if m:
            d[int(m.group(1))] = m.group(2).strip().upper()
    return d


def llm_scores(tname, model):
    """{k: [acc per seed]} for a model's ans dir."""
    out = {}
    for fn, meta in man['tasks'][tname]['files'].items():
        ans = f"{B}/{tname}/ans/{model}/{os.path.basename(fn).replace('.txt', '.ans')}"
        if not os.path.exists(ans):
            continue
        pred = parse(ans)
        tt = {i: str(l).upper() for i, l in meta['truth']}
        if not pred:
            continue
        acc = sum(1 for i, l in tt.items() if pred.get(i) == l) / len(tt)
        out.setdefault(meta['k'], []).append(acc)
    return out


for tname, t in man['tasks'].items():
    ch = t['chance']; lr = t['logreg']
    opus = llm_scores(tname, 'opus')
    qwen = llm_scores(tname, 'qwen')
    print(f"\n=== {tname.upper()} (chance {ch:.3f}) — graphlex+LLM vs logreg, mean+/-std ===")
    print(f"  {'k':>3} {'Opus(3s)':>14} {'Qwen14b(8s)':>16} {'logreg':>14}")
    for k in man['k_logreg']:
        lv = lr.get(str(k), [])
        lrc = f"{np.mean(lv):.3f}+/-{np.std(lv):.3f}" if lv else "       -"
        oc = (f"{np.mean(opus[k]):.3f}+/-{np.std(opus[k]):.3f}"
              if k in opus and opus[k] else "       -")
        qc = (f"{np.mean(qwen[k]):.3f}+/-{np.std(qwen[k]):.3f}"
              if k in qwen and qwen[k] else "       -")
        print(f"  {k:>3} {oc:>14} {qc:>16} {lrc:>14}")
