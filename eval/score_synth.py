"""Score subagent answers in synth_v2/ans/<model>/<stem>.ans against manifest truth.

Answer files: one line per query, '<id> <TOKEN>' (robust regex parse).
Reports per-(model,arm) mean accuracy +/- std across seeds, plus logreg ref.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_ans as parse, raw_acc

BASE = '/home/scratch/bench_out/synth_v2'
man = json.load(open(f"{BASE}/manifest.json"))
chance = man["chance"]


def score_file(ansfile, truth):
    return raw_acc(truth, parse(ansfile))


# discover models = subdirs of ans/
models = sorted(d for d in os.listdir(f"{BASE}/ans")
                if os.path.isdir(f"{BASE}/ans/{d}"))
arms = ['raw', 'raw_perm', 'counts_only', 'verbal', 'verbal_anon']

print(f"chance = {chance:.3f}\n")
lr = list(man["logreg"].values())
print(f"logreg(features) ref : {np.mean(lr):.3f} +/- {np.std(lr):.3f}  "
      f"(n_seed={len(lr)}, nq={man['nq']}/seed)\n")

rows = []
for model in models:
    for arm in arms:
        accs = []
        for fn, meta in man["files"].items():
            if meta["arm"] != arm:
                continue
            ans = f"{BASE}/ans/{model}/{fn.replace('.txt', '.ans')}"
            if os.path.exists(ans):
                a = score_file(ans, meta["truth"])
                if a is not None:
                    accs.append(a)
        if accs:
            rows.append((model, arm, np.mean(accs), np.std(accs), len(accs)))

print(f"{'model':12} {'arm':12} {'mean':>6} {'std':>6} {'nseed':>6}")
print("-" * 46)
for model, arm, mu, sd, n in rows:
    print(f"{model:12} {arm:12} {mu:6.3f} {sd:6.3f} {n:6d}")

json.dump([{"model": m, "arm": a, "mean": mu, "std": sd, "nseed": n}
           for m, a, mu, sd, n in rows],
          open(f"{BASE}/scores.json", 'w'), indent=0)
print(f"\n-> {BASE}/scores.json")
