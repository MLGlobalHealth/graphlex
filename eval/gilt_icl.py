"""GILT native-in-context whole-graph classification at the SAME matched few-shot
splits as the graphlex sweep (comparison #2: pretrained native-ICL graph FM).

GILT (arXiv 2510.04567, "GILT: An LLM-Free, Tuning-Free Graph Foundational Model
for In-Context Learning", code github.com/yiming421/inductnode, checkpoint
huggingface.co/fdsajkshf/gilt-checkpoint/gilt_model.pt) is a tuning-free graph
foundation model: given K labeled SUPPORT graphs/class + query graphs, it predicts
every query in ONE no-gradient forward pass via a PFN transformer over per-graph
prototype embeddings. This is the closest non-LLM analogue to graphlex+LLM: both
do few-shot whole-graph classification with no per-task gradient step. We run it
head-to-head on the IDENTICAL shots/splits/seeds and score BALANCED accuracy.

----------------------------------------------------------------------------
HOW GILT INGESTS A GRAPH (verified against the repo's single-task GC path,
src/engine_gc.py:evaluate_graph_classification_single_task @3066,
_create_context_embeddings_computed @810, _safe_lookup_node_embeddings @447):

  * The GNN (PureGCN_v1, 6 layers, hidden=128) is a near-parameter-free linear
    GCN; its INPUT node features must be 128-dim float, L2-row-normalized. All
    learned capacity is in the PFN transformer + GC head.
  * Featurization we apply (documented, dataset-wide, fit on support+query nodes):
      - clean one-hot categorical node feats (== _common.node_cats, the SAME node
        type info the LLM/logreg arms use via comp()/'type') -> use them as raw;
      - else (featureless: IMDB / social / ego-net sets) -> degree one-hot
        (clipped to MAX_DEG), the standard TU fallback (matches gnn_baseline.py).
      Then: PCA to 128 if raw_dim>=128, else PCA to raw_dim and ZERO-PAD to 128
      (GILT's padding_strategy='zero'); finally F.normalize(p=2,dim=1). This
      reproduces data_gc.process_graph_features (PCA->pad->L2) exactly.
  * Pooling is MAX (graph_pooling='max' in the checkpoint args), not mean.
  * We bypass GILT's FUG external-mapping by putting the 128-d features directly
    in each graph's .x (float 2D, width>1): _safe_lookup_node_embeddings Case 2
    returns x as-is when no 'fug_mapping' is present. dataset.node_embs is set to
    a placeholder so _get_node_embedding_table doesn't raise.

----------------------------------------------------------------------------
MATCHED-SPLIT PROTOCOL (byte-for-byte sweep.py / fm_repr_baseline.py /
gnn_baseline.py):
    ds   = TUDataset(TU_ROOT, name); cap = min(len(ds), POOL_CAP)
    idx  = [i<cap : ds[i].num_nodes>=3 and ds[i].edge_index.size(1)>=1]
    spc  = max(2, min(SHOTS_PER_CLASS, MAX_SHOTS // n_classes))
    per seed in SEEDS:
        rng=RandomState(seed); pos[c]=rng.permutation(positions y==c)
        shot+=pos[c][:spc]; q+=pos[c][spc:]; rng.shuffle(q); q=q[:NQ]; rng.shuffle(shot)
We assert the reconstructed query truth == the manifest's stored truth
(SystemExit on mismatch), exactly like the other arms -> provably apples-to-apples.

Output: $OUT/<dataset>.json, keyed model='gilt', mirroring fm_repr/gnn keying:
    {"dataset","model":"gilt","spc","n_classes","n_query","seeds":[...],
     "feat":..., "results":{"gilt":{seed:balacc}}, "mean":{"gilt":[mu,sd]}, "config":{...}}

Run (clpc35, gpfn_venv + torch_sparse, with the inductnode repo on sys.path):
    GILT_REPO=/home/scratch/graphlex_icl/inductnode \
    GILT_CKPT=/home/scratch/graphlex_icl/inductnode/checkpoints/gilt_model.pt \
    python eval/gilt_icl.py --datasets MUTAG --seeds 11
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Batch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _common import bal_acc, node_cats  # noqa: E402

# Split constants -- MUST stay in lockstep with sweep.py / fm_repr_baseline.py.
SEEDS = [11, 22, 33]
SHOTS_PER_CLASS = 5
MAX_SHOTS = 60
NQ = 40
POOL_CAP = 4000

# Featurization
HIDDEN = 128            # GILT input dim == hidden (model.lin is Identity)
MAX_DEG = 50            # degree one-hot cap for featureless datasets (== gnn_baseline)
POOLING = "max"         # graph_pooling in the checkpoint args


# ============================================================================
# GILT model reconstruction (patches the repo's missing-arg gap)
# ============================================================================
def load_gilt(repo, ckpt, device):
    """Rebuild GILT model + PFN predictor from gilt_model.pt.

    recreate_model_from_checkpoint reads args.degree directly, but the released
    checkpoint's saved args omit 'degree' -> AttributeError. We inject the
    config default (None) into the checkpoint's args dict before reconstruction.
    """
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import src.checkpoint_utils as cu  # noqa: E402
    import src.model as gmodel          # noqa: E402

    # Read the saved args so we can supply head-depth params the builder omits.
    info, _ = cu.load_checkpoint_config(ckpt)
    saved = info["args"]
    head_kw = {
        "head_num_layers": saved.get("head_num_layers", 2),
        "gc_head_num_layers": saved.get("gc_head_num_layers", None),
        "nc_head_num_layers": saved.get("nc_head_num_layers", None),
        "lp_head_num_layers": saved.get("lp_head_num_layers", None),
        "gc_sim": saved.get("gc_sim", "dot"),
        "gc_ridge_alpha": saved.get("gc_ridge_alpha", 1.0),
    }

    # --- Patch 1: the builder reads args.degree without getattr; the released
    #     checkpoint omits it -> inject the config default (None). ---
    orig_load_cfg = cu.load_checkpoint_config

    def patched_load_cfg(path):
        i, c = orig_load_cfg(path)
        i["args"].setdefault("degree", None)
        return i, c

    # --- Patch 2: recreate_model_from_checkpoint constructs PFNPredictorNodeCls
    #     WITHOUT passing head_num_layers/gc_head_num_layers, so gc_head defaults
    #     to a 3-layer MLP while the checkpoint's gc_head is a single linear
    #     (gc_head_num_layers=1) -> 'Missing keys' + randomly-initialised head.
    #     We wrap the ctor to inject the saved head-depth params, then verify a
    #     CLEAN load (strict, zero missing/unexpected keys). ---
    PredCls = gmodel.PFNPredictorNodeCls
    orig_init = PredCls.__init__

    def patched_init(self, *a, **kw):
        for k, v in head_kw.items():
            kw.setdefault(k, v)
        return orig_init(self, *a, **kw)

    cu.load_checkpoint_config = patched_load_cfg
    PredCls.__init__ = patched_init
    try:
        model, predictor, att, mlp, proj, idproj, args_dict = \
            cu.recreate_model_from_checkpoint(ckpt, HIDDEN, device)
        # Verify a clean predictor load now (no randomly-initialised head).
        psd = torch.load(ckpt, map_location="cpu", weights_only=False)[
            "predictor_state_dict"]
        miss, unexp = predictor.load_state_dict(psd, strict=False)
        miss = [k for k in miss if "num_batches_tracked" not in k]
        if miss or unexp:
            raise SystemExit(
                f"GILT predictor did NOT load cleanly after head-depth patch: "
                f"missing={miss[:8]} unexpected={list(unexp)[:8]}. "
                f"Refusing to score with a partially-random model.")
    finally:
        cu.load_checkpoint_config = orig_load_cfg
        PredCls.__init__ = orig_init
    model.eval()
    predictor.eval()
    return model, predictor


# ============================================================================
# Matched splits (mirror sweep.py / fm_repr_baseline.py)
# ============================================================================
def load_idx_y(ds):
    cap = min(len(ds), POOL_CAP)
    idx = [i for i in range(cap)
           if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx])
    return idx, y


def split_for_seed(idx, y, classes, spc, seed):
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


# ============================================================================
# Featurization -> 128-d L2-normalized node features (GILT's expected input)
# ============================================================================
def dataset_uses_node_cats(ds, idx):
    for j in idx[: min(50, len(idx))]:
        cats, ncat = node_cats(ds[j])
        if cats is not None and ncat > 0:
            return True
    return False


def raw_node_features(data, use_cats, ncat):
    """Raw per-node feature matrix [n, d] before PCA: one-hot node category if the
    dataset is clean-categorical, else degree one-hot."""
    n = data.num_nodes
    ei = data.edge_index
    if use_cats:
        cats, nc = node_cats(data)
        x = torch.zeros(n, ncat)
        if cats is not None and nc > 0:
            x[torch.arange(n), torch.as_tensor(cats, dtype=torch.long)] = 1.0
        return x
    # degree one-hot
    deg = torch.zeros(n, dtype=torch.long)
    if ei.numel() > 0:
        deg = torch.bincount(ei[0], minlength=n)[:n]
    deg = deg.clamp(max=MAX_DEG)
    x = torch.zeros(n, MAX_DEG + 1)
    x[torch.arange(n), deg] = 1.0
    return x


def to_128(raw_list):
    """PCA->128 (or PCA->d then zero-pad to 128), then L2-normalize each node row.
    PCA is fit jointly over ALL nodes of all support+query graphs (matches
    data_gc.process_graph_features, which fits over the passed dataset)."""
    stacked = np.concatenate([r.numpy() for r in raw_list], axis=0)  # [Ntot, d]
    d = stacked.shape[1]
    if d >= HIDDEN:
        k = HIDDEN
        pca = PCA(n_components=k, svd_solver="auto", random_state=0)
        red = pca.fit_transform(stacked)
        out = red
    else:
        k = min(d, stacked.shape[0])
        pca = PCA(n_components=k, svd_solver="auto", random_state=0)
        red = pca.fit_transform(stacked)
        out = np.zeros((stacked.shape[0], HIDDEN), dtype=np.float32)
        out[:, :k] = red
    out = torch.tensor(out, dtype=torch.float32)
    out = F.normalize(out, p=2, dim=1)
    # split back per graph
    feats, off = [], 0
    for r in raw_list:
        nn = r.shape[0]
        feats.append(out[off:off + nn])
        off += nn
    return feats


def make_data(ds_graph, x128):
    """A PyG Data whose .x is the 128-d L2-normalized feature (float 2D)."""
    return Data(x=x128, edge_index=ds_graph.edge_index, y=ds_graph.y.view(-1))


# ============================================================================
# Minimal dataset_info wrapper (no fug_mapping; x carries the features directly)
# ============================================================================
class _ListDataset:
    """Indexable dataset over a fixed list of Data; carries num_classes + a
    placeholder node_embs so _get_node_embedding_table doesn't raise."""
    def __init__(self, graphs, num_classes):
        self._g = graphs
        self.num_classes = num_classes
        self.name = "graphlex"
        # placeholder; never indexed because x is already a float 2D matrix
        self.node_embs = torch.zeros(1, HIDDEN)

    def __len__(self):
        return len(self._g)

    def __getitem__(self, i):
        return self._g[i]


class _QueryLoader:
    """Yields a single Batch of all query graphs (one forward pass). Exposes
    .dataset for the engine's len() check."""
    def __init__(self, graphs):
        self.dataset = graphs
        self._graphs = graphs

    def __iter__(self):
        yield Batch.from_data_list(self._graphs)


# ============================================================================
# Per-dataset driver
# ============================================================================
def run_dataset(name, tu_root, model, predictor, device, seeds,
                manifest=None, verbose=True):
    from src.engine_gc import evaluate_graph_classification_single_task
    from argparse import Namespace

    ds = TUDataset(tu_root, name=name)
    idx, y = load_idx_y(ds)
    classes = sorted(set(y.tolist()))
    cls_index = {c: k for k, c in enumerate(classes)}  # contiguous 0..C-1 labels
    spc = max(2, min(SHOTS_PER_CLASS, MAX_SHOTS // len(classes)))
    use_cats = dataset_uses_node_cats(ds, idx)
    feat_kind = "node-cats(one-hot)->PCA128" if use_cats else f"deg-onehot(<= {MAX_DEG})->PCA128"

    # ncat for one-hot width (max over a sample, like gnn/sweep)
    ncat = 0
    if use_cats:
        for j in idx:
            _, nc = node_cats(ds[j])
            ncat = max(ncat, nc)

    if verbose:
        print(f"[{name}] graphs={len(idx)} classes={len(classes)} spc={spc} "
              f"NQ<= {NQ} feat={feat_kind} pool={POOLING}", flush=True)

    gc_args = Namespace(gc_metric="accuracy", use_graph_cs=False,
                        gc_train_eval_max_batches=0)

    results = {"gilt": {}}
    nq_seen = {}
    for seed in seeds:
        shot, q = split_for_seed(idx, y, classes, spc, seed)
        ys_shot = y[shot]
        yq = y[q]
        nq_seen[seed] = len(q)

        # parity: reconstructed query truth must match manifest
        if manifest is not None:
            mf = manifest.get("files", {}).get(f"{name}/seed{seed}.txt")
            if mf is not None:
                recon = [f"CLASS{int(yq[i])}".upper() for i in range(len(yq))]
                stored = [str(lab).upper() for _, lab in mf["truth"]]
                if recon != stored:
                    raise SystemExit(
                        f"SPLIT MISMATCH {name} seed{seed}: reconstructed query "
                        f"truth != manifest truth (len {len(recon)} vs {len(stored)}). "
                        f"GILT split does not match the LLM arm.")

        # featurize shots+queries jointly (dataset-wide PCA over their nodes)
        all_pos = list(shot) + list(q)
        raw = [raw_node_features(ds[idx[j]], use_cats, ncat) for j in all_pos]
        feats = to_128(raw)
        shot_graphs = [make_data(ds[idx[all_pos[k]]], feats[k])
                       for k in range(len(shot))]
        q_graphs = [make_data(ds[idx[all_pos[len(shot) + k]]], feats[len(shot) + k])
                    for k in range(len(q))]

        dataset_obj = _ListDataset(shot_graphs + q_graphs, len(classes))

        # context_graphs = {task0: {class_idx: {'graphs':[...], 'indices':[...]}}}
        ctx = {0: {}}
        for k, j in enumerate(shot):
            c = cls_index[int(ys_shot[k])]
            ctx[0].setdefault(c, {"graphs": [], "indices": []})
            ctx[0][c]["graphs"].append(shot_graphs[k])
            ctx[0][c]["indices"].append(k)  # position in dataset_obj

        dataset_info = {
            "dataset": dataset_obj,
            "context_graphs": ctx,
            "num_classes": len(classes),
            "needs_identity_projection": False,
        }
        data_loaders = {"test": _QueryLoader(q_graphs)}

        with torch.no_grad():
            res = evaluate_graph_classification_single_task(
                model, predictor, dataset_info, data_loaders, task_idx=0,
                pooling_method=POOLING, device=device, normalize_class_h=True,
                dataset_name=name, identity_projection=None, args=gc_args,
                return_logits=True)

        logits = res["test_logits"]            # [NQ, C], order matches q_graphs
        pred_idx = logits.argmax(dim=1).numpy()
        # map predicted class-index back to original class label
        inv = {k: c for c, k in cls_index.items()}
        pred_lab = np.array([inv[int(p)] for p in pred_idx])

        tl = [(i, f"CLASS{int(yq[i])}") for i in range(len(yq))]
        pm = {i: f"CLASS{int(pred_lab[i])}".upper() for i in range(len(pred_lab))}
        ba = bal_acc(tl, pm)
        ba = float(ba) if ba is not None else 0.0
        results["gilt"][str(seed)] = ba
        if verbose:
            print(f"  seed{seed} gilt balacc={ba:.3f} (NQ={len(yq)})", flush=True)

    vals = [results["gilt"][str(s)] for s in seeds if str(s) in results["gilt"]]
    mean = {"gilt": ([float(np.mean(vals)), float(np.std(vals))] if vals else None)}

    return {
        "dataset": name,
        "model": "gilt",
        "spc": spc,
        "n_classes": len(classes),
        "n_query": nq_seen,
        "seeds": list(seeds),
        "feat": feat_kind,
        "results": results,
        "mean": mean,
        "config": {
            "checkpoint": "fdsajkshf/gilt-checkpoint/gilt_model.pt",
            "gnn": "PureGCN_v1(6L,h128)", "predictor": "PFN",
            "pooling": POOLING, "input_dim": HIDDEN, "max_deg": MAX_DEG,
            "shots_per_class": SHOTS_PER_CLASS, "max_shots": MAX_SHOTS,
            "nq": NQ, "pool_cap": POOL_CAP,
            "featurize": "node-cats|deg-onehot -> PCA/zeropad to 128 -> L2norm",
        },
    }


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma list; default=manifest")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--tu-root", default=os.environ.get("TU_ROOT", "/home/scratch/tudata"))
    ap.add_argument("--out", default=os.environ.get("GILT_OUT", "/home/scratch/bench_out/gilt"))
    ap.add_argument("--manifest", default=os.environ.get(
        "SWEEP_MANIFEST", "/home/scratch/graphlex_icl/manifest.json"))
    ap.add_argument("--repo", default=os.environ.get(
        "GILT_REPO", "/home/scratch/graphlex_icl/inductnode"))
    ap.add_argument("--ckpt", default=os.environ.get(
        "GILT_CKPT", "/home/scratch/graphlex_icl/inductnode/checkpoints/gilt_model.pt"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = json.load(open(args.manifest)) if os.path.exists(args.manifest) else None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    elif manifest is not None:
        datasets = sorted(manifest["baselines"].keys())
    else:
        raise SystemExit("no --datasets and no manifest")

    print(f"device={device} datasets={len(datasets)} seeds={seeds} out={args.out}", flush=True)
    print(f"loading GILT from {args.ckpt}", flush=True)
    model, predictor = load_gilt(args.repo, args.ckpt, device)

    done, failed = [], []
    for name in datasets:
        out_path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"SKIP {name} (exists)", flush=True)
            done.append(name)
            continue
        t0 = time.time()
        try:
            res = run_dataset(name, args.tu_root, model, predictor, device,
                              seeds, manifest=manifest)
            json.dump(res, open(out_path, "w"), indent=1)
            m = res["mean"]["gilt"]
            print(f"DONE {name} ({time.time()-t0:.1f}s) "
                  f"balacc={m[0]:.3f}+-{m[1]:.3f} -> {out_path}", flush=True)
            done.append(name)
        except SystemExit:
            raise
        except Exception as e:
            import traceback
            print(f"ERR {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            traceback.print_exc()
            failed.append((name, f"{type(e).__name__}: {str(e)[:160]}"))

    print(f"\nALL DONE: {len(done)} ok, {len(failed)} failed", flush=True)
    for n, e in failed:
        print(f"  FAILED {n}: {e}", flush=True)


if __name__ == "__main__":
    main()
