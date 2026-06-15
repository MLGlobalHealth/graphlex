"""Trained GNN baselines (GIN + GCN, graph classification) at the SAME matched
few-shot splits as the graphlex sweep.

This fills the gap in the sweep: every existing arm is either an LLM reasoning
over verbalize(facts(G)), a logreg on the facts() feature vector, or majority.
There is NO *trained* GNN. This script adds few-shot GIN and GCN so the
label-efficiency story has the obvious specialist bar: a GNN trained on the
exact same K labeled graphs/class, evaluated on the exact same query graphs,
scored with the exact same BALANCED accuracy.

Matched-split protocol (byte-for-byte the sweep / balanced_rescore.py logic):
    ds   = TUDataset(TU_ROOT, name)
    cap  = min(len(ds), POOL_CAP)
    idx  = [i<cap : ds[i].num_nodes>=3 and ds[i].edge_index.size(1)>=1]
    y    = labels over idx
    spc  = max(2, min(SHOTS_PER_CLASS, MAX_SHOTS // n_classes))
    per seed in SEEDS:
        rng = RandomState(seed)
        pos[c] = rng.permutation(positions with y==c)
        shot += pos[c][:spc]; q += pos[c][spc:]
        rng.shuffle(q); q = q[:NQ]; rng.shuffle(shot)
The shot positions feed GNN training (with a held-out slice for early stopping)
and q positions are the queries. We assert the reconstructed query truth matches
the manifest's stored truth so the split is provably identical to the LLM arm's.

Node featurization (consistent, documented):
  * If data.x is a clean one-hot categorical (same test sweep/_common.node_cats
    uses), keep it as-is -> the GNN sees the SAME node-type information the
    classical/LLM arms get via comp()/'type' attrs.
  * Else (no node features, e.g. IMDB / social / many ego-net datasets), use a
    degree-based one-hot feature (degree clipped to MAX_DEG, one-hot), which is
    the standard TU fallback (cf. PyG benchmark scripts / GIN paper appendix).
  All graphs in a dataset are featurized with a single, dataset-wide scheme.

Hyperparameters are FIXED across all 30 datasets (no per-dataset tuning); see
GNN_BASELINE_PLAN.md. The few-shot GNN is EXPECTED to struggle on this tiny
label budget -- that is the legitimate, honest result for the label-efficiency
story, not something to tune away.

Output: a results JSON per dataset at $OUT/<dataset>.json with structure
    {"dataset","spc","n_classes","n_query","seeds":[...],
     "results": {"gin": {seed: balacc, ...}, "gcn": {...}},
     "mean": {"gin": [mu, sd], "gcn": [mu, sd]},
     "config": {...}}
mirroring the sweep's per-dataset/seed/model keying so make_figures.py can fold
GIN/GCN in as new heatmap arms.

Run (smoke, on the cluster venv):
    python eval/gnn_baseline.py --datasets MUTAG,IMDB-BINARY --seeds 11 --epochs 40
Full sweep is driven by gnn_baseline.slurm (array over datasets).
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data

# Reuse the sweep's split constants + the balanced-accuracy metric so the GNN
# arm is provably apples-to-apples with the LLM / logreg / majority arms.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _common import bal_acc, node_cats  # noqa: E402

# Split constants copied (not imported) from sweep.py so this file runs without
# the bench_out paths / graphlex import that sweep.py pulls at module load.
# These MUST stay in lockstep with sweep.py / balanced_rescore.py.
SEEDS = [11, 22, 33]
SHOTS_PER_CLASS = 5
MAX_SHOTS = 60
NQ = 40
POOL_CAP = 4000

# --- fixed GNN config (no per-dataset tuning) --------------------------------
HIDDEN = 64
N_LAYERS = 4
DROPOUT = 0.5
LR = 1e-2
WEIGHT_DECAY = 5e-4
EPOCHS = 200            # cap; early stopping usually fires far earlier
PATIENCE = 30           # early-stop patience on the held-out shot slice
VAL_FRAC = 0.25         # fraction of shots held out for early stopping
MAX_DEG = 50            # degree one-hot cap for featureless datasets
BATCH = 32


# ============================================================================
# Matched splits (mirror sweep.py / balanced_rescore.py exactly)
# ============================================================================
def load_idx_y(ds):
    cap = min(len(ds), POOL_CAP)
    idx = [i for i in range(cap)
           if ds[i].num_nodes >= 3 and ds[i].edge_index.size(1) >= 1]
    y = np.array([int(ds[i].y) for i in idx])
    return idx, y


def split_for_seed(idx, y, classes, spc, seed):
    """Return (shot, q) position lists -- identical to sweep.py."""
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
# Featurization
# ============================================================================
def degree_onehot(edge_index, n, max_deg=MAX_DEG):
    deg = torch.zeros(n, dtype=torch.long)
    if edge_index.numel() > 0:
        deg = torch.bincount(edge_index[0], minlength=n)[:n]
    deg = deg.clamp(max=max_deg)
    x = torch.zeros(n, max_deg + 1)
    x[torch.arange(n), deg] = 1.0
    return x


def dataset_uses_node_cats(ds, idx):
    """Decide once per dataset whether to keep one-hot node features (clean
    categorical, matching _common.node_cats) or fall back to degree one-hot."""
    # sample a handful of graphs; node_cats returns (None,0) if not clean one-hot
    for j in idx[: min(50, len(idx))]:
        cats, ncat = node_cats(ds[j])
        if cats is not None and ncat > 0:
            return True
    return False


def featurize(data, use_cats):
    n = data.num_nodes
    ei = data.edge_index
    if use_cats and data.x is not None:
        cats, ncat = node_cats(data)
        if cats is not None and ncat > 0:
            x = torch.zeros(n, ncat)
            x[torch.arange(n), torch.as_tensor(cats, dtype=torch.long)] = 1.0
        else:
            # this graph isn't clean one-hot though the dataset mostly is;
            # fall back to a single constant column padded to ncat width.
            x = torch.zeros(n, 1)
    else:
        x = degree_onehot(ei, n)
    return Data(x=x, edge_index=ei, y=data.y)


def build_graphs(ds, positions, idx, use_cats):
    return [featurize(ds[idx[j]], use_cats) for j in positions]


# ============================================================================
# Models
# ============================================================================
class GIN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            d = in_dim if i == 0 else hidden
            mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            d = in_dim if i == 0 else hidden
            self.convs.append(GCNConv(d, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


MODELS = {"gin": GIN, "gcn": GCN}


# ============================================================================
# Train / eval one (model, seed)
# ============================================================================
def stratified_val_split(ys, classes, seed, val_frac=VAL_FRAC):
    """Hold out a per-class slice of the shots for early stopping. With very few
    shots/class we keep >=1 train and >=1 val per class where possible."""
    rng = np.random.RandomState(seed + 7)
    train_local, val_local = [], []
    for c in classes:
        ci = [k for k in range(len(ys)) if ys[k] == c]
        rng.shuffle(ci)
        nv = max(1, int(round(len(ci) * val_frac))) if len(ci) >= 2 else 0
        val_local += ci[:nv]
        train_local += ci[nv:]
    if not train_local:                # degenerate; train on everything
        train_local = list(range(len(ys)))
        val_local = list(range(len(ys)))
    if not val_local:
        val_local = list(train_local)
    return train_local, val_local


def run_one(model_name, in_dim, n_classes, shot_graphs, ys_shot, q_graphs,
            yq, classes, seed, device, epochs, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_local, va_local = stratified_val_split(ys_shot, classes, seed)
    tr_graphs = [shot_graphs[k] for k in tr_local]
    va_graphs = [shot_graphs[k] for k in va_local]

    tr_loader = DataLoader(tr_graphs, batch_size=min(BATCH, len(tr_graphs)),
                           shuffle=True)
    va_loader = DataLoader(va_graphs, batch_size=BATCH)
    q_loader = DataLoader(q_graphs, batch_size=BATCH)

    model = MODELS[model_name](in_dim, HIDDEN, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # class weights to counter shot imbalance (many-class datasets get spc capped)
    cw = torch.ones(n_classes, device=device)
    bc = np.bincount(ys_shot, minlength=n_classes).astype(float)
    cw = torch.tensor(np.where(bc > 0, bc.sum() / (len(bc) * bc), 1.0),
                      dtype=torch.float, device=device)
    crit = nn.CrossEntropyLoss(weight=cw)

    best_val, best_state, since = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for b in tr_loader:
            b = b.to(device)
            opt.zero_grad()
            out = model(b.x, b.edge_index, b.batch)
            loss = crit(out, b.y.view(-1))
            loss.backward()
            opt.step()
        # early stop on held-out shot slice (balanced acc)
        va = eval_balacc(model, va_loader, device)
        if va > best_val:
            best_val, since = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # query balanced accuracy
    qba = eval_balacc(model, q_loader, device, truth_y=yq)
    if verbose:
        print(f"    {model_name}: best_val_balacc={best_val:.3f} "
              f"query_balacc={qba:.3f} (epochs_run<= {ep + 1})", flush=True)
    return qba


@torch.no_grad()
def eval_balacc(model, loader, device, truth_y=None):
    model.eval()
    preds, trues = [], []
    for b in loader:
        b = b.to(device)
        out = model(b.x, b.edge_index, b.batch)
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(b.y.view(-1).cpu().numpy())
    if not preds:
        return 0.0
    preds = np.concatenate(preds)
    trues = np.concatenate(trues) if truth_y is None else np.asarray(truth_y)
    # use the shared balanced-accuracy metric for exact parity with the sweep
    tl = [(i, f"CLASS{int(trues[i])}") for i in range(len(trues))]
    pm = {i: f"CLASS{int(preds[i])}".upper() for i in range(len(preds))}
    ba = bal_acc(tl, pm)
    return float(ba) if ba is not None else 0.0


# ============================================================================
# Per-dataset driver
# ============================================================================
def run_dataset(name, tu_root, seeds, device, epochs, manifest=None, verbose=True):
    ds = TUDataset(tu_root, name=name)
    idx, y = load_idx_y(ds)
    classes = sorted(set(y.tolist()))
    spc = max(2, min(SHOTS_PER_CLASS, MAX_SHOTS // len(classes)))
    use_cats = dataset_uses_node_cats(ds, idx)

    feat_kind = "node-cats(one-hot)" if use_cats else f"degree-onehot(<= {MAX_DEG})"
    if verbose:
        print(f"[{name}] graphs={len(idx)} classes={len(classes)} spc={spc} "
              f"NQ<= {NQ} feat={feat_kind}", flush=True)

    results = {m: {} for m in MODELS}
    nq_seen = {}
    for seed in seeds:
        shot, q = split_for_seed(idx, y, classes, spc, seed)
        ys_shot = y[shot]
        yq = y[q]
        nq_seen[seed] = len(q)

        # provable parity: reconstructed query truth must match the manifest
        if manifest is not None:
            mf = manifest.get("files", {}).get(f"{name}/seed{seed}.txt")
            if mf is not None:
                man_truth = [f"CLASS{int(yq[i])}" for i in range(len(yq))]
                stored = [str(lab).upper() for _, lab in mf["truth"]]
                if [t.upper() for t in man_truth] != stored:
                    raise SystemExit(
                        f"SPLIT MISMATCH for {name} seed{seed}: reconstructed "
                        f"query truth != manifest truth (len {len(man_truth)} vs "
                        f"{len(stored)}). The GNN split does not match the LLM arm.")

        # featurize shots+queries with the dataset-wide scheme
        shot_graphs = build_graphs(ds, shot, idx, use_cats)
        q_graphs = build_graphs(ds, q, idx, use_cats)
        in_dim = shot_graphs[0].x.size(1)

        for m in MODELS:
            t0 = time.time()
            qba = run_one(m, in_dim, len(classes), shot_graphs, ys_shot,
                          q_graphs, yq, classes, seed, device, epochs,
                          verbose=verbose)
            results[m][str(seed)] = qba
            if verbose:
                print(f"  seed{seed} {m:3} balacc={qba:.3f} "
                      f"({time.time() - t0:.1f}s)", flush=True)

    mean = {}
    for m in MODELS:
        vals = [results[m][str(s)] for s in seeds if str(s) in results[m]]
        mean[m] = [float(np.mean(vals)), float(np.std(vals))] if vals else None

    return {
        "dataset": name,
        "spc": spc,
        "n_classes": len(classes),
        "n_query": nq_seen,
        "seeds": list(seeds),
        "feat": feat_kind,
        "results": results,
        "mean": mean,
        "config": {
            "hidden": HIDDEN, "n_layers": N_LAYERS, "dropout": DROPOUT,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs_cap": epochs,
            "patience": PATIENCE, "val_frac": VAL_FRAC, "batch": BATCH,
            "max_deg": MAX_DEG, "pool": {"gin": "add", "gcn": "mean"},
            "shots_per_class": SHOTS_PER_CLASS, "max_shots": MAX_SHOTS,
            "nq": NQ, "pool_cap": POOL_CAP,
        },
    }


# ============================================================================
# Full-supervision reference (SECONDARY; clearly a different setting)
# ============================================================================
def run_full_supervision(name, tu_root, device, epochs, seeds, verbose=True):
    """Standard 80/10/10 train/val/test GIN as an upper-bound reference row.
    NOT matched few-shot -- marked separately in the output ('setting':'full')."""
    ds = TUDataset(tu_root, name=name)
    idx, y = load_idx_y(ds)
    classes = sorted(set(y.tolist()))
    use_cats = dataset_uses_node_cats(ds, idx)
    accs = {"gin": []}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(idx))
        ntr, nva = int(0.8 * len(perm)), int(0.1 * len(perm))
        tr, va, te = perm[:ntr], perm[ntr:ntr + nva], perm[ntr + nva:]
        gtr = [featurize(ds[idx[k]], use_cats) for k in tr]
        gte = [featurize(ds[idx[k]], use_cats) for k in te]
        in_dim = gtr[0].x.size(1)
        ys_tr = y[tr]
        qba = run_one("gin", in_dim, len(classes), gtr, ys_tr, gte,
                      y[te], classes, seed, device, epochs)
        accs["gin"].append(qba)
        if verbose:
            print(f"  [full] {name} seed{seed} gin test_balacc={qba:.3f}", flush=True)
    return {"dataset": name, "setting": "full",
            "mean": {"gin": [float(np.mean(accs["gin"])), float(np.std(accs["gin"]))]},
            "results": {"gin": {str(s): a for s, a in zip(seeds, accs["gin"])}}}


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None,
                    help="comma list; default = all in manifest")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tu-root", default=os.environ.get(
        "TU_ROOT", "/home/scratch/tudata"))
    ap.add_argument("--out", default=os.environ.get(
        "GNN_OUT", "/home/scratch/bench_out/gnn"))
    ap.add_argument("--manifest", default=os.environ.get(
        "SWEEP_MANIFEST", "/home/scratch/bench_out/sweep/manifest.json"))
    ap.add_argument("--full-supervision", action="store_true",
                    help="also run the 80/10/10 full-supervision GIN reference")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if result JSON exists")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = None
    if os.path.exists(args.manifest):
        manifest = json.load(open(args.manifest))

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    elif manifest is not None:
        datasets = sorted(manifest["baselines"].keys())
    else:
        raise SystemExit("no --datasets and no manifest to enumerate from")

    print(f"device={device} datasets={len(datasets)} seeds={seeds} "
          f"epochs<= {args.epochs} out={args.out}", flush=True)
    print(f"GNN config: hidden={HIDDEN} layers={N_LAYERS} dropout={DROPOUT} "
          f"lr={LR} wd={WEIGHT_DECAY} patience={PATIENCE}", flush=True)

    for name in datasets:
        out_path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"SKIP {name} (exists: {out_path})", flush=True)
            continue
        t0 = time.time()
        try:
            res = run_dataset(name, args.tu_root, seeds, device, args.epochs,
                              manifest=manifest)
            if args.full_supervision:
                res["full_supervision"] = run_full_supervision(
                    name, args.tu_root, device, args.epochs, seeds)
            json.dump(res, open(out_path, "w"), indent=1)
            gm = res["mean"].get("gin")
            cm = res["mean"].get("gcn")
            print(f"DONE {name} ({time.time() - t0:.1f}s) "
                  f"gin={gm[0]:.3f} gcn={cm[0]:.3f} -> {out_path}", flush=True)
        except SystemExit:
            raise
        except Exception as e:
            print(f"ERR {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
