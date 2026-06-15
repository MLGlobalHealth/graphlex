"""Foundation-model representation baseline (frozen embeddings + logreg head) at
the SAME matched few-shot splits as the graphlex sweep.

This adds the "representation mode" arm for graph foundation models: instead of
asking an LLM to reason over verbalize(facts(G)) or training a GNN end-to-end,
we take a pretrained graph FM's FROZEN graph-level embedding for each graph and
fit a tiny LogisticRegression head on the same K labeled graphs/class, then
predict the same query graphs. Same split, same BALANCED accuracy as every other
arm -> provably apples-to-apples.

Currently wired for the GraphPFN encoder (encoder='graphpfn'). The embeddings are
precomputed graph-level pooled vectors at:
    /home/scratch/fm_embeddings_sweep/graphpfn__{NAME}.npz   key 'pooled' (N_sel,768)
where row j of 'pooled' corresponds to the j-th SELECTED graph under the sweep's
selection rule:
    ds   = TUDataset(TU_ROOT, name)
    cap  = min(len(ds), POOL_CAP)
    idx  = [i<cap : ds[i].num_nodes>=3 and ds[i].edge_index.size(1)>=1]
i.e. pooled[j] <-> ds[idx[j]]. So we reconstruct the sweep's (shot, q) POSITION
lists (which index into idx) and use them directly to slice 'pooled'.

Matched-split protocol (byte-for-byte the sweep / balanced_rescore.py / gnn_baseline.py):
    spc  = max(2, min(SHOTS_PER_CLASS, MAX_SHOTS // n_classes))
    per seed in SEEDS:
        rng = RandomState(seed)
        pos[c] = rng.permutation(positions with y==c)
        shot += pos[c][:spc]; q += pos[c][spc:]
        rng.shuffle(q); q = q[:NQ]; rng.shuffle(shot)
We assert the reconstructed query truth matches the manifest's stored truth
(SystemExit on mismatch) before scoring, exactly like gnn_baseline.py.

Head: StandardScaler -> LogisticRegression(max_iter=5000), matching the convention
in eval/crossdomain_graphcls.py for FM-embedding logreg heads.

Output: a results JSON per dataset at $OUT/<dataset>.json with structure
    {"dataset","encoder","spc","n_classes","n_query","seeds":[...],
     "results": {"graphpfn": {seed: balacc, ...}},
     "mean": {"graphpfn": [mu, sd]}, "config": {...}}
mirroring the sweep / gnn_baseline.py keying so make_figures.py can fold the
FM-embed arm in as a new heatmap row.

Run (CPU is fine; logreg on 768-d is fast):
    python eval/fm_repr_baseline.py
    python eval/fm_repr_baseline.py --datasets MUTAG,IMDB-BINARY --seeds 11
"""
import os
import sys
import json
import time
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import TUDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _common import bal_acc  # noqa: E402

# Split constants -- MUST stay in lockstep with sweep.py / balanced_rescore.py /
# gnn_baseline.py.
SEEDS = [11, 22, 33]
SHOTS_PER_CLASS = 5
MAX_SHOTS = 60
NQ = 40
POOL_CAP = 4000

# Encoder -> embedding-file template. Add kumorfm/gmn here once sweep-wide npz
# coverage exists; today only graphpfn covers all 30 sweep datasets.
EMB_DIRS = ["/home/scratch/fm_embeddings_sweep", "/home/scratch/real_fm_embeddings"]
DEFAULT_ENCODER = "graphpfn"
EMB_KEY = "pooled"

LOGREG_MAX_ITER = 5000  # matches eval/crossdomain_graphcls.py


# ============================================================================
# Matched splits (mirror sweep.py / balanced_rescore.py / gnn_baseline.py)
# ============================================================================
def load_idx_y(ds):
    cap = min(len(ds), POOL_CAP)
    idx = [i for i in range(cap)
           if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx])
    return idx, y


def split_for_seed(idx, y, classes, spc, seed):
    """Return (shot, q) POSITION lists (index into idx) -- identical to sweep.py."""
    rng = np.random.RandomState(seed)
    pos = {c: list(rng.permutation([j for j in range(len(idx)) if y[j] == c]))
           for c in classes}
    shot, q = [], []
    for c in classes:
        shot += pos[c][:spc]
        q += pos[c][spc:]
    rng.shuffle(q)
    q = q[:NQ]
    rng.shuffle(shot)
    return shot, q


def find_emb(encoder, name):
    """Locate the npz embedding file for (encoder, dataset name)."""
    for d in EMB_DIRS:
        p = os.path.join(d, f"{encoder}__{name}.npz")
        if os.path.exists(p):
            return p
    return None


# ============================================================================
# Per-dataset driver
# ============================================================================
def run_dataset(name, tu_root, encoder, seeds, manifest=None, verbose=True):
    emb_path = find_emb(encoder, name)
    if emb_path is None:
        raise FileNotFoundError(
            f"no {encoder} embedding npz for {name} in {EMB_DIRS}")

    pooled = np.load(emb_path)[EMB_KEY]
    if not np.isfinite(pooled).all():
        raise ValueError(f"{emb_path}: non-finite embeddings")

    ds = TUDataset(tu_root, name=name)
    idx, y = load_idx_y(ds)
    classes = sorted(set(y.tolist()))
    spc = max(2, min(SHOTS_PER_CLASS, MAX_SHOTS // len(classes)))

    # Row alignment: pooled[j] <-> ds[idx[j]] (the j-th selected graph).
    if pooled.shape[0] != len(idx):
        raise ValueError(
            f"{name}: embedding rows {pooled.shape[0]} != selected graphs "
            f"{len(idx)} -- selection rule mismatch; rows do not align.")

    if verbose:
        print(f"[{name}] enc={encoder} graphs={len(idx)} dim={pooled.shape[1]} "
              f"classes={len(classes)} spc={spc} NQ<= {NQ}", flush=True)

    results = {encoder: {}}
    nq_seen = {}
    for seed in seeds:
        shot, q = split_for_seed(idx, y, classes, spc, seed)
        ys_shot = y[shot]
        yq = y[q]
        nq_seen[seed] = len(q)

        # provable parity: reconstructed query truth must match the manifest,
        # exactly like gnn_baseline.py.
        if manifest is not None:
            mf = manifest.get("files", {}).get(f"{name}/seed{seed}.txt")
            if mf is not None:
                recon = [f"CLASS{int(yq[i])}".upper() for i in range(len(yq))]
                stored = [str(lab).upper() for _, lab in mf["truth"]]
                if recon != stored:
                    raise SystemExit(
                        f"SPLIT MISMATCH for {name} seed{seed}: reconstructed "
                        f"query truth != manifest truth (len {len(recon)} vs "
                        f"{len(stored)}). The FM-embed split does not match the "
                        f"LLM arm.")

        # slice the embedding rows for shot/query graphs (positions index idx,
        # and pooled is aligned to idx)
        Xs = pooled[shot]
        Xq = pooled[q]

        sc = StandardScaler().fit(Xs)
        clf = LogisticRegression(max_iter=LOGREG_MAX_ITER).fit(
            sc.transform(Xs), ys_shot)
        pc = clf.predict(sc.transform(Xq))

        # BALANCED accuracy via the shared metric (exact parity with sweep arms)
        tl = [(i, f"CLASS{int(yq[i])}") for i in range(len(yq))]
        pm = {i: f"CLASS{int(pc[i])}".upper() for i in range(len(pc))}
        ba = bal_acc(tl, pm)
        ba = float(ba) if ba is not None else 0.0
        results[encoder][str(seed)] = ba
        if verbose:
            print(f"  seed{seed} balacc={ba:.3f}", flush=True)

    vals = [results[encoder][str(s)] for s in seeds if str(s) in results[encoder]]
    mean = {encoder: ([float(np.mean(vals)), float(np.std(vals))] if vals else None)}

    return {
        "dataset": name,
        "encoder": encoder,
        "spc": spc,
        "n_classes": len(classes),
        "n_query": nq_seen,
        "seeds": list(seeds),
        "emb_path": emb_path,
        "emb_dim": int(pooled.shape[1]),
        "results": results,
        "mean": mean,
        "config": {
            "head": "StandardScaler+LogisticRegression",
            "logreg_max_iter": LOGREG_MAX_ITER,
            "shots_per_class": SHOTS_PER_CLASS, "max_shots": MAX_SHOTS,
            "nq": NQ, "pool_cap": POOL_CAP, "emb_key": EMB_KEY,
        },
    }


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None,
                    help="comma list; default = all in manifest")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--tu-root", default=os.environ.get(
        "TU_ROOT", "/home/scratch/tudata"))
    ap.add_argument("--out", default=os.environ.get(
        "FM_REPR_OUT", "/home/scratch/bench_out/fm_repr"))
    ap.add_argument("--manifest", default=os.environ.get(
        "SWEEP_MANIFEST", "/home/scratch/bench_out/sweep/manifest.json"))
    ap.add_argument("--force", action="store_true",
                    help="recompute even if result JSON exists")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.encoder)
    os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    manifest = None
    if os.path.exists(args.manifest):
        manifest = json.load(open(args.manifest))

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    elif manifest is not None:
        datasets = sorted(manifest["baselines"].keys())
    else:
        raise SystemExit("no --datasets and no manifest to enumerate from")

    print(f"encoder={args.encoder} datasets={len(datasets)} seeds={seeds} "
          f"out={out_dir}", flush=True)

    done, failed = [], []
    for name in datasets:
        out_path = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"SKIP {name} (exists: {out_path})", flush=True)
            done.append(name)
            continue
        t0 = time.time()
        try:
            res = run_dataset(name, args.tu_root, args.encoder, seeds,
                              manifest=manifest)
            json.dump(res, open(out_path, "w"), indent=1)
            m = res["mean"][args.encoder]
            print(f"DONE {name} ({time.time() - t0:.1f}s) "
                  f"balacc={m[0]:.3f}+-{m[1]:.3f} -> {out_path}", flush=True)
            done.append(name)
        except SystemExit:
            raise
        except Exception as e:
            print(f"ERR {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            failed.append((name, f"{type(e).__name__}: {str(e)[:160]}"))

    print(f"\nALL DONE: {len(done)} ok, {len(failed)} failed", flush=True)
    if failed:
        for n, e in failed:
            print(f"  FAILED {n}: {e}", flush=True)


if __name__ == "__main__":
    main()
