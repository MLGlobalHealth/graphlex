"""Score the broad cross-domain sweep: graphlex+LLM (Qwen all-seeds, Opus subset)
vs classical+logreg vs majority, per dataset, grouped by domain, with regret vs the
best non-LLM baseline (classical/majority). Reports a flexibility summary."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_ans as parse

B = '/home/scratch/bench_out/sweep'
man = json.load(open(f"{B}/manifest.json"))


def llm(ds, model):
    accs = []
    for fn, meta in man['files'].items():
        if meta['dataset'] != ds:
            continue
        a = f"{B}/{ds}/ans/{model}/seed{meta['seed']}.ans"
        if not os.path.exists(a):
            continue
        pred = parse(a)
        tt = {i: str(l).upper() for i, l in meta['truth']}
        if pred:
            accs.append(sum(1 for i, l in tt.items() if pred.get(i) == l) / len(tt))
    return accs


rows = []
for ds, meta in sorted(man['meta'].items(), key=lambda kv: (kv[1]['domain'], kv[0])):
    b = man['baselines'][ds]
    cl, mj = b['classical'][0], b['majority'][0]
    qw = llm(ds, 'qwen'); op = llm(ds, 'opus')
    rows.append((meta['domain'], ds, meta['chance'], cl, mj,
                 np.mean(qw) if qw else None, np.mean(op) if op else None))

hdr = f"{'domain':12} {'dataset':16} {'chance':>7} {'classical':>9} {'major':>7} {'Qwen14b':>8} {'Opus':>7} {'Qregret':>8}"
print(hdr); print('-' * len(hdr))
qreg = []
for dom, ds, ch, cl, mj, qw, op in rows:
    best = max(cl, mj)  # best non-LLM baseline
    qr = (best - qw) if qw is not None else None
    if qr is not None:
        qreg.append(qr)
    f = lambda x: f"{x:.3f}" if x is not None else "   -"
    qrs = f"{qr:+.3f}" if qr is not None else "   -"
    print(f"{dom:12} {ds:16} {ch:7.3f} {cl:9.3f} {mj:7.3f} {f(qw):>8} {f(op):>7} {qrs:>8}")

def summary(name, getter):
    regs = []
    for dom, ds, ch, cl, mj, qw, op in rows:
        v = getter(qw, op)
        if v is not None:
            regs.append(max(cl, mj) - v)
    if not regs:
        return
    r = np.array(regs)
    print(f"\n=== flexibility: {name} vs best non-LLM baseline ===")
    print(f"  scored {len(r)} datasets / {len(set(x[0] for x in rows))} domains | "
          f"mean regret {r.mean():+.3f} | median {np.median(r):+.3f} | worst {r.max():+.3f}")
    print(f"  >= baseline (regret<=0): {int((r<=0).sum())}/{len(r)} | "
          f"within 0.05: {int((r<=0.05).sum())}/{len(r)} | "
          f"substantially worse (>0.10): {int((r>0.10).sum())}/{len(r)}")


summary("Qwen-14b (3 seeds)", lambda qw, op: qw)
summary("Opus (seed11)", lambda qw, op: op)
