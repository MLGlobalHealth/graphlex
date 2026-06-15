"""Score the graph-level REGRESSION ICL track: graphlex+LLM (Opus + Qwen) vs the
Ridge / GNN / predict-the-mean baselines, MAE / RMSE / R^2, mean over seeds.

Unlike the classification tracks (score_node_icl.py etc.), the answers here are
REAL NUMBERS, not class tokens, so _common.parse_ans does NOT apply. We use a
TOLERANT NUMERIC parser (parse_num) that pulls '<id> <number>' lines in all the
near-format variants the models emit:
    "0 -0.73"   "Query 0: -0.73"   "0) 1.2"   "0 - -0.5"   "0  z=-0.73"
    "0: -1.4e0"   "0\t2"   (also tolerates a trailing unit word after the number)
The LLM predicts a STANDARDIZED z-score (the prompt shows standardized targets);
we de-standardize with the per-file zmean/zstd from the manifest BEFORE computing
metrics, so LLM MAE/RMSE/R^2 are in the SAME raw units as the baselines.

Run: /home/scratch/fmsn-dev/.venv/bin/python eval/score_regression_icl.py [DATASET]
"""
import os, sys, re, json
import numpy as np

B = '/home/scratch/bench_out/regression_icl'

# tolerant numeric answer line: leading id, optional 'query', optional separator,
# optional 'z=' / '=' noise, then a signed float (int / decimal / scientific).
NUM_LINE = re.compile(
    r'^\s*(?:query\s*)?(\d+)\s*[:.\)\-]?\s*(?:z\s*=\s*|=\s*)?'
    r'(?:[\-−]\s*)?'                      # optional separating dash before value
    r'([+\-−]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*', re.I)


def parse_num(path):
    """Parse an .ans file -> {int query id: float prediction}. Tolerant; ignores
    unparseable lines. Handles a stray '-' separator between id and value by taking
    the LAST signed float on the line as the prediction."""
    d = {}
    with open(path) as fh:
        for ln in fh:
            s = ln.strip().replace('−', '-')   # normalize unicode minus
            m = re.match(r'^\s*(?:query\s*)?(\d+)\b', s, re.I)
            if not m:
                continue
            qid = int(m.group(1))
            rest = s[m.end():]
            # take the last signed float in the remainder (robust to "0 - -0.5",
            # "0: z=-0.73", "0) 1.2 kcal/mol")
            nums = re.findall(r'[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?', rest)
            if not nums:
                continue
            try:
                d[qid] = float(nums[-1])
            except ValueError:
                continue
    return d


def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float('nan')
    return mae, rmse, r2


def llm_scores(man, model):
    """{k: [(mae,rmse,r2) per seed]} for a model's ans dir. De-standardizes the
    z-score predictions back to raw units with each file's zmean/zstd."""
    out = {}
    for fn, meta in man['files'].items():
        ans = (f"{B}/{os.path.dirname(fn)}/ans/{model}/"
               f"{os.path.basename(fn).replace('.txt', '.ans')}")
        if not os.path.exists(ans):
            continue
        pred = parse_num(ans)
        if not pred:
            continue
        zmean, zstd = meta['zmean'], meta['zstd']
        yt, yp = [], []
        for qi, raw in meta['truth']:
            if qi in pred:
                yt.append(raw)
                yp.append(pred[qi] * zstd + zmean)   # de-standardize
        if len(yt) < 2:
            continue
        out.setdefault(meta['k'], []).append((reg_metrics(yt, yp), len(yt)))
    return out


def _bm(agg_d):
    """manifest baseline block {metric:[vals]} -> 'MAE x RMSE y R2 z' (mean)."""
    if not agg_d or not agg_d.get('mae'):
        return "            -"
    return (f"MAE {np.mean(agg_d['mae']):.3f} RMSE {np.mean(agg_d['rmse']):.3f} "
            f"R2 {np.mean(agg_d['r2']):.3f}")


def _lm(vals):
    """[((mae,rmse,r2), ncov), ...] -> 'MAE x RMSE y R2 z (cov c)'."""
    if not vals:
        return "            -"
    mets = np.array([v[0] for v in vals])
    cov = int(np.mean([v[1] for v in vals]))
    return (f"MAE {mets[:,0].mean():.3f} RMSE {mets[:,1].mean():.3f} "
            f"R2 {mets[:,2].mean():.3f} (cov {cov})")


def main(dataset):
    base = f"{B}/{dataset}"
    man = json.load(open(f"{base}/manifest.json"))
    ts = man['target_stats']
    print(f"\n=== {dataset} graph-REGRESSION ICL  (target: {man['target']}) ===")
    print(f"  {man['n_graphs']} graphs; target mean {ts['mean']:.2f} std {ts['std']:.2f}; "
          f"nq={man['nq']}; seeds={man['seeds']}; metrics in RAW units, mean over seeds")
    print(f"  -- baselines --")
    print(f"  {'K':>3} {'predict-mean':>34} {'Ridge(facts)':>34} {'GNN few-shot':>34}")
    for k in man['k_shots']:
        ks = str(k)
        print(f"  {k:>3} {_bm(man['mean_baseline'].get(ks)):>34} "
              f"{_bm(man['ridge'].get(ks)):>34} {_bm(man.get('gnn_fewshot',{}).get(ks)):>34}")
    if man.get('gnn_full'):
        print(f"  GNN full-supervision (train {man['train_full']}, upper bar): "
              f"{_bm(man['gnn_full'])}")
    # LLM arms
    opus = llm_scores(man, 'opus')
    qwen = llm_scores(man, 'qwen')
    print(f"  -- graphlex+LLM (de-standardized to raw units; cov = queries parsed) --")
    print(f"  {'K':>3} {'Opus':>42} {'Qwen':>42}")
    for k in man['k_shots']:
        print(f"  {k:>3} {_lm(opus.get(k, [])):>42} {_lm(qwen.get(k, [])):>42}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'FreeSolv')
