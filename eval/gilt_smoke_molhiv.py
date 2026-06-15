"""GILT smoke test: reproduce the paper's few-shot ogbg-molhiv graph-classification
AUC with the released checkpoint, using GILT's OWN featurization recipe.

Paper reports ~58.17 AUC (5-shot) on molhiv. GILT featurizes molecules with the
RAW OGB 9-dim atom features -> PCA to 128 -> L2-normalize (src/data_fug.py
load_ogb_original_features + data_gc.process_graph_features), MAX graph pooling,
PFN predictor (dot sim). We rebuild the model with eval/gilt_icl.load_gilt (which
patches the repo's missing-arg + head-depth bugs and verifies a CLEAN state load),
then run evaluate_graph_classification_single_task on the OGB test split with K
balanced support graphs sampled from the OGB train split. AUC over several seeds.

If this lands near ~0.58 the checkpoint + our wiring are validated; if it is at
chance (~0.5) the checkpoint or our wiring is broken and GILT should be dropped.

Run:
    GILT_REPO=/home/scratch/graphlex_icl/inductnode \
    python eval/gilt_smoke_molhiv.py --shots 5 --queries 500 --seeds 11,22,33
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, Batch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gilt_icl as G  # reuse loader + featurization + dataset wrappers


def featurize_ogb(graphs):
    """Raw 9-dim OGB atom features -> PCA128 -> L2norm (GILT's recipe), per the
    same to_128 used for TU sets, fit jointly over all passed graphs' nodes."""
    raw = [g.x.float() for g in graphs]   # OGB x is int64 atom features
    feats = G.to_128(raw)
    out = []
    for g, f in zip(graphs, feats):
        out.append(Data(x=f, edge_index=g.edge_index, y=g.y.view(-1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ogbg-molhiv")
    ap.add_argument("--shots", type=int, default=5, help="support graphs per class")
    ap.add_argument("--queries", type=int, default=500, help="cap test queries (AUC)")
    ap.add_argument("--seeds", default="11,22,33")
    ap.add_argument("--ogb-root", default=os.environ.get("OGB_ROOT", "/home/scratch/ogb"))
    ap.add_argument("--repo", default=os.environ.get(
        "GILT_REPO", "/home/scratch/graphlex_icl/inductnode"))
    ap.add_argument("--ckpt", default=os.environ.get(
        "GILT_CKPT", "/home/scratch/graphlex_icl/inductnode/checkpoints/gilt_model.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    from ogb.graphproppred import PygGraphPropPredDataset
    ds = PygGraphPropPredDataset(name=args.dataset, root=args.ogb_root)
    split = ds.get_idx_split()
    train_idx = split["train"].numpy()
    test_idx = split["test"].numpy()
    y = np.array([int(ds[int(i)].y.view(-1)[0]) for i in range(len(ds))])

    print(f"{args.dataset}: {len(ds)} graphs; train={len(train_idx)} "
          f"test={len(test_idx)}; pos-rate(test)="
          f"{y[test_idx].mean():.3f}", flush=True)

    model, predictor = G.load_gilt(args.repo, args.ckpt, device)
    from src.engine_gc import evaluate_graph_classification_single_task
    from argparse import Namespace
    gc_args = Namespace(gc_metric="auc", use_graph_cs=False,
                        gc_train_eval_max_batches=0)

    aucs = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        # K balanced support graphs/class from TRAIN split
        sup = []
        for c in [0, 1]:
            pool = train_idx[y[train_idx] == c]
            sup += list(rng.choice(pool, size=args.shots, replace=False))
        # query graphs from TEST split (balanced cap for a stable AUC)
        qpos = test_idx[y[test_idx] == 1]
        qneg = test_idx[y[test_idx] == 0]
        nq = min(args.queries // 2, len(qpos), len(qneg))
        qry = list(rng.choice(qpos, nq, replace=False)) + \
              list(rng.choice(qneg, nq, replace=False))
        rng.shuffle(qry)

        sup_raw = [ds[int(i)] for i in sup]
        qry_raw = [ds[int(i)] for i in qry]
        # featurize support+query jointly (PCA fit over their nodes, GILT-style)
        all_g = featurize_ogb(sup_raw + qry_raw)
        sup_g = all_g[: len(sup)]
        qry_g = all_g[len(sup):]
        ys_sup = np.array([int(g.y[0]) for g in sup_g])
        yq = np.array([int(g.y[0]) for g in qry_g])

        dataset_obj = G._ListDataset(sup_g + qry_g, 2)
        ctx = {0: {}}
        for k in range(len(sup_g)):
            c = int(ys_sup[k])
            ctx[0].setdefault(c, {"graphs": [], "indices": []})
            ctx[0][c]["graphs"].append(sup_g[k])
            ctx[0][c]["indices"].append(k)
        dataset_info = {"dataset": dataset_obj, "context_graphs": ctx,
                        "num_classes": 2, "needs_identity_projection": False}
        data_loaders = {"test": G._QueryLoader(qry_g)}

        with torch.no_grad():
            res = evaluate_graph_classification_single_task(
                model, predictor, dataset_info, data_loaders, task_idx=0,
                pooling_method=G.POOLING, device=device, normalize_class_h=True,
                dataset_name=args.dataset, identity_projection=None, args=gc_args,
                return_logits=True)
        logits = res["test_logits"]            # [NQ, 2]
        prob_pos = F.softmax(logits, dim=1)[:, 1].numpy()
        auc = roc_auc_score(yq, prob_pos)
        aucs.append(auc)
        print(f"  seed{seed}: {args.shots}-shot AUC={auc:.4f} (NQ={len(yq)})", flush=True)

    print(f"\n{args.dataset} {args.shots}-shot AUC: "
          f"mean={np.mean(aucs):.4f} +- {np.std(aucs):.4f} "
          f"over seeds {seeds}", flush=True)
    print(f"PAPER reference (molhiv 5-shot): ~0.5817", flush=True)
    print(json.dumps({"dataset": args.dataset, "shots": args.shots,
                      "aucs": aucs, "mean": float(np.mean(aucs))}))


if __name__ == "__main__":
    main()
