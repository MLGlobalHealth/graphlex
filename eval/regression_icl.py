"""Graph-level REGRESSION few-shot ICL track: graphlex+LLM vs Ridge-on-facts vs a
GNN regressor vs predict-the-mean, on a MoleculeNet continuous target.

FOURTH variant of the taxonomy. After whole-graph CLASSIFICATION (sweep.py /
label_curve.py), node classification (node_icl.py), and edge/link prediction
(edge_icl.py) -- all CATEGORICAL -- this points the SAME facts()/verbalize()
machinery at a CONTINUOUS graph-level target: predict a real number per graph.
Each ICL "example" = one verbalized molecule -> its measured value; the query =
a molecule, predict its number. This completes the granularity x output-type grid
(graph/node/edge  x  classification/regression). See REGRESSION_TRACK_PLAN.md.

Mirrors node_icl.py / edge_icl.py:
  * pure-ICL: writes seed*_k*.txt prompt files in the EXIST ING drivers' format
    (run_qwen.py / run_opus_cli.py); this script calls NO LLM. Same skeleton: TASK
    header, "=== LABELED EXAMPLES ===" with [<value>] blocks, "=== QUERIES ===",
    "Query <id>:" blocks. OUTPUT FORMAT is "<id> <number>" (a REAL number, not a
    class token) -- the numeric parser lives in score_regression_icl.py.
  * K-shot support: K example molecules (graph -> value pairs), nested in K.
  * BASELINES written into manifest.json (all FREE, no LLM):
      (1) Ridge regression on the graphlex fact-vector (_common.fvec) -- the
          classical bar, SAME features the LLM sees, at the SAME K shots.
      (2) GNN regressor (GIN + global mean-pool + linear head, MSE) -- few-shot
          (trained on the K shots, expected near-useless at K=1..few -> reported
          honestly) AND a full-supervision GNN (trained on a large train split,
          eval on the query set) as the specialist upper bar, parallel to the
          node-track transductive GCN.
      (3) predict-the-mean -- the trivial regression floor (predict the train
          mean for every query; R^2=0 by construction on the train distribution).
  * METRICS: MAE, RMSE, R^2 (regression, NOT balanced accuracy).

TARGET STANDARDIZATION: the LLM prompt shows STANDARDIZED targets (z-scores using
the train-shot mean/std) so the numbers are O(1) and unit-free; the prompt states
the mean/std so values can be de-standardized at scoring time. Ridge/GNN/mean
baselines are computed on the RAW target and reported in RAW units (kcal/mol for
FreeSolv) -- the manifest records the standardization so the scorer maps LLM
z-score predictions back to raw units before computing MAE/RMSE/R^2.

VERBALIZATION: each molecule -> networkx graph with a 'type' node attribute =
element symbol (from MoleculeNet atom-feature col 0 = atomic number) ->
graphlex facts()/verbalize(node_attrs='type'). graphlex core is NOT modified.

Run:  /home/scratch/fmsn-dev/.venv/bin/python eval/regression_icl.py [DATASET]
      DATASET in {FreeSolv, ESOL, Lipo}; default FreeSolv. Env SMOKE=1 -> tiny grid.
"""
import os, sys, json
import numpy as np
import networkx as nx
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import to_networkx
import torch

sys.path.insert(0, '/home/scratch/Dropbox/Seth/Research/MLGHrepos/graphlex')
from graphlex import facts, verbalize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import fvec   # graphlex facts() -> fixed-length feature vector

MOLNET_ROOT = '/home/scratch/molnet'
OUT_BASE = '/home/scratch/bench_out/regression_icl'

# dataset -> (MoleculeNet name, human target name + unit)
DATASETS = {
    'FreeSolv': ('FreeSolv', 'hydration free energy (kcal/mol)'),
    'ESOL': ('ESOL', 'log-solubility (log mol/L)'),
    'Lipo': ('Lipophilicity', 'octanol/water logD'),
}

# atomic number -> element symbol (the elements present in these MoleculeNet sets)
Z2SYM = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 15: 'P',
         16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}

SEEDS = [11, 22, 33]          # >=3 seeds (load-bearing)
K_SHOTS = [1, 3, 5, 10]       # example molecules (nested in K)
NQ = 12                       # query molecules (kept small so a smoke LLM pass is cheap)
TRAIN_FULL = 400              # train pool for the full-supervision GNN upper bar
GNN_REP = 'facts'             # only the facts/element verbalization is shown to LLM

# 'facts' rep: graphlex structural verbalization + element composition line.
REPS = ['facts']

if os.environ.get('SMOKE'):
    SEEDS, K_SHOTS, NQ, TRAIN_FULL = [11], [1, 3], 12, 400


# --- molecule -> networkx with element-typed nodes ----------------------------
def mol_nx(data):
    """PyG molecule Data -> undirected networkx graph with a 'type' node attribute
    = element symbol (from atom-feature col 0 = atomic number)."""
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    z = data.x[:, 0].tolist()
    nx.set_node_attributes(
        G, {i: Z2SYM.get(int(zi), f"Z{int(zi)}") for i, zi in enumerate(z)}, 'type')
    return G


def element_line(data):
    """Readable element-composition line appended to the verbalization (parallel to
    node_icl's per-node readable line). graphlex core untouched."""
    z = data.x[:, 0].tolist()
    syms = [Z2SYM.get(int(zi), f"Z{int(zi)}") for zi in z]
    from collections import Counter
    c = Counter(syms)
    comp = ", ".join(f"{s}x{n}" for s, n in sorted(c.items(), key=lambda kv: -kv[1]))
    return f"Atoms ({len(syms)} total): {comp}"


def verbalize_mol(data):
    """Verbalize a molecule: graphlex structural facts (node_attrs='type') + a
    readable element-composition line."""
    G = mol_nx(data)
    struct = verbalize(facts(G, node_attrs='type'), focus='structure')
    return f"Molecule graph: {struct}\n{element_line(data)}"


# --- K-shot + query splits ----------------------------------------------------
def make_splits(n, seeds, k_max, nq, train_full):
    """Per seed: a permutation -> k_max SHOT ids (nested in K) + nq QUERY ids
    (disjoint) + a TRAIN_FULL pool for the full-supervision GNN (disjoint from
    queries; may overlap the shots, which is fine -- it's the specialist bar)."""
    out = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        shots = perm[:k_max].tolist()
        queries = perm[k_max:k_max + nq].tolist()
        used = set(queries)
        full = [i for i in perm.tolist() if i not in used][:train_full]
        out[seed] = {"shots": shots, "queries": queries, "full": full}
    return out


# --- regression metrics -------------------------------------------------------
def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float('nan')
    return {"mae": mae, "rmse": rmse, "r2": r2}


# --- baseline 1: Ridge on the graphlex fact-vector ----------------------------
def ridge_at(feats, y, shots, queries, k):
    """Ridge regression on the graphlex fact-vector, trained on K shots, eval on
    the query molecules. SAME features the LLM sees. Returns metrics dict."""
    tr = shots[:k]
    Xtr = np.array([feats[i] for i in tr]); ytr = y[tr]
    Xq = np.array([feats[i] for i in queries]); yq = y[queries]
    if len(tr) < 2:                       # K=1: Ridge degenerates -> predict the shot
        pred = np.full(len(queries), float(ytr[0]))
        return reg_metrics(yq, pred)
    sc = StandardScaler().fit(Xtr)
    clf = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xq))
    return reg_metrics(yq, pred)


# --- baseline 3: predict-the-mean floor ---------------------------------------
def mean_at(y, shots, queries, k):
    """Predict the train-shot mean for every query (trivial regression floor)."""
    tr = shots[:k]
    pred = np.full(len(queries), float(np.mean(y[tr])))
    return reg_metrics(y[queries], pred)


# --- baseline 2: GNN regressor (GIN + global mean-pool + linear head) ---------
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader

GNN_HIDDEN = 64
GNN_LAYERS = 3
GNN_LR = 1e-2
GNN_WD = 5e-4
GNN_EPOCHS = 200
GNN_PATIENCE = 30


class GINReg(nn.Module):
    """GIN encoder + global mean-pool + linear head -> scalar (MSE regression)."""
    def __init__(self, in_dim, hidden=GNN_HIDDEN, n_layers=GNN_LAYERS):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            d = in_dim if i == 0 else hidden
            mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.head(global_mean_pool(x, batch)).view(-1)


def _to_float_data(ds, ids, y, ymean, ystd):
    """Build float-featured PyG graphs (standardized x) with standardized target."""
    from torch_geometric.data import Data
    out = []
    for i in ids:
        d = ds[int(i)]
        x = d.x.float()
        out.append(Data(x=x, edge_index=d.edge_index,
                        y=torch.tensor([(float(y[i]) - ymean) / ystd], dtype=torch.float)))
    return out


def gnn_at(ds, y, train_ids, queries, seed, label):
    """Train GINReg on train_ids (standardized target), eval on the query molecules.
    Used both for the few-shot bar (train_ids = K shots) and the full-supervision
    upper bar (train_ids = TRAIN_FULL pool). Returns metrics in RAW target units."""
    if len(train_ids) < 2:
        return None
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ytr_raw = y[train_ids]
    ymean, ystd = float(ytr_raw.mean()), float(ytr_raw.std() + 1e-8)
    tr = _to_float_data(ds, train_ids, y, ymean, ystd)
    qg = _to_float_data(ds, queries, y, ymean, ystd)
    in_dim = tr[0].x.size(1)
    model = GINReg(in_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WD)
    loader = DataLoader(tr, batch_size=min(32, len(tr)), shuffle=True)
    best, best_state, since = 1e9, None, 0
    for ep in range(GNN_EPOCHS):
        model.train()
        tot = 0.0
        for b in loader:
            b = b.to(dev); opt.zero_grad()
            out = model(b.x, b.edge_index, b.batch)
            loss = F.mse_loss(out, b.y)
            loss.backward(); opt.step()
            tot += float(loss.detach()) * b.num_graphs
        tot /= len(tr)
        if tot < best - 1e-4:
            best, since = tot, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= GNN_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    qloader = DataLoader(qg, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for b in qloader:
            b = b.to(dev)
            preds.append(model(b.x, b.edge_index, b.batch).cpu().numpy())
    pred_z = np.concatenate(preds)
    pred_raw = pred_z * ystd + ymean        # de-standardize to raw units
    return reg_metrics(y[queries], pred_raw)


# --- prompt builder -----------------------------------------------------------
def build_prompt(ds, y, shots, queries, k, target_name, zmean, zstd):
    """One ICL regression prompt for a (seed,k). Targets are shown STANDARDIZED
    (z = (raw - zmean)/zstd, where zmean/zstd are the K-shot mean/std). The prompt
    states zmean/zstd so the scorer can de-standardize. truth holds RAW values."""
    TASK = (
        f"Each item is a MOLECULE shown as its graph structure (atoms = nodes, bonds "
        f"= edges) plus its element composition. Predict a CONTINUOUS property: the "
        f"{target_name}.\n"
        f"Values are given as STANDARDIZED z-scores (z = (value - {zmean:.4f}) / "
        f"{zstd:.4f}); predict the z-score for each query (a real number, typically "
        f"in [-3, 3]). Learn the structure->value relationship from the labeled "
        f"examples, then predict each query.\n"
        f"OUTPUT FORMAT: one line per query, exactly '<id> <number>' where <number> "
        f"is the predicted z-score (e.g. '0 -0.73'). No other text.")
    L = [TASK, "", "=== LABELED EXAMPLES ==="]
    for i in shots[:k]:
        z = (float(y[i]) - zmean) / zstd
        L.append(f"[value z={z:.3f}]\n{verbalize_mol(ds[int(i)])}\n")
    L.append("=== QUERIES (predict each) ===")
    truth = []
    for qi, i in enumerate(queries):
        L.append(f"Query {qi}:\n{verbalize_mol(ds[int(i)])}\n")
        truth.append([qi, float(y[i])])          # RAW value (scorer de-standardizes pred)
    return "\n".join(L), truth


def run(dataset):
    name, target_name = DATASETS[dataset]
    out = f"{OUT_BASE}/{dataset}"
    os.makedirs(out, exist_ok=True)
    ds = MoleculeNet(MOLNET_ROOT, name=name)
    n = len(ds)
    y = np.array([float(ds[i].y.view(-1)[0]) for i in range(n)])

    # graphlex fact-vectors once (Ridge baseline + the same features the LLM sees)
    feats = [fvec(facts(mol_nx(ds[i]))) for i in range(n)]

    yglobstd = float(y.std() + 1e-8)      # fallback scale when shot std ~0 (K=1)

    splits = make_splits(n, SEEDS, max(K_SHOTS), NQ, TRAIN_FULL)

    # baselines (free): Ridge + mean per (seed,k); few-shot GNN per (seed,k);
    # full-supervision GNN once per seed (K-independent specialist bar).
    ridge = {str(k): [] for k in K_SHOTS}
    meanb = {str(k): [] for k in K_SHOTS}
    gnn_fs = {str(k): [] for k in K_SHOTS}
    gnn_full = []
    for seed in SEEDS:
        sh = splits[seed]["shots"]; q = splits[seed]["queries"]; full = splits[seed]["full"]
        gf = gnn_at(ds, y, full, q, seed, "full")
        if gf is not None:
            gnn_full.append(gf)
        for k in K_SHOTS:
            ridge[str(k)].append(ridge_at(feats, y, sh, q, k))
            meanb[str(k)].append(mean_at(y, sh, q, k))
            g = gnn_at(ds, y, sh[:k], q, seed, f"fs{k}")
            if g is not None:
                gnn_fs[str(k)].append(g)

    # prompt files
    rep = REPS[0]
    os.makedirs(f"{out}/{rep}/ans/opus", exist_ok=True)
    os.makedirs(f"{out}/{rep}/ans/qwen", exist_ok=True)
    files = {}
    for seed in SEEDS:
        sh = splits[seed]["shots"]; q = splits[seed]["queries"]
        for k in K_SHOTS:
            tr = sh[:k]
            zmean = float(np.mean(y[tr]))
            # shot std is the natural scale, but at K=1 (or all-equal shots) it is ~0,
            # which makes the z-score prompt meaningless -> fall back to the global
            # target std as the scale (the scorer uses the SAME zmean/zstd to invert).
            sstd = float(np.std(y[tr]))
            zstd = sstd if sstd > 1e-3 else yglobstd
            prompt, truth = build_prompt(ds, y, sh, q, k, target_name, zmean, zstd)
            fn = f"{dataset}/{rep}/seed{seed}_k{k}.txt"
            open(f"{OUT_BASE}/{fn}", 'w').write(prompt)
            files[fn] = {"dataset": dataset, "rep": rep, "seed": seed, "k": k,
                         "truth": truth, "query_ids": [int(i) for i in q],
                         "support_ids": [int(i) for i in tr],
                         "zmean": zmean, "zstd": zstd}

    def agg(metric_list):
        """[ {mae,rmse,r2}, ... ] -> {metric: [vals]} for the manifest."""
        out_m = {"mae": [], "rmse": [], "r2": []}
        for m in metric_list:
            for kk in out_m:
                out_m[kk].append(m[kk])
        return out_m

    man = {
        "dataset": dataset, "task": "graph_regression", "moleculenet_name": name,
        "target": target_name, "seeds": SEEDS, "k_shots": K_SHOTS, "nq": NQ,
        "n_graphs": n, "train_full": TRAIN_FULL, "representations": REPS,
        "target_stats": {"min": float(y.min()), "max": float(y.max()),
                         "mean": float(y.mean()), "std": float(y.std())},
        "standardization": "per-(seed,k) z-score on the K shots; zmean/zstd in each file entry",
        "files": files,
        "ridge": {k: agg(v) for k, v in ridge.items()},
        "mean_baseline": {k: agg(v) for k, v in meanb.items()},
        "gnn_fewshot": {k: agg(v) for k, v in gnn_fs.items() if v},
        "gnn_full": agg(gnn_full) if gnn_full else None,
        "metric": "MAE / RMSE / R2 (raw target units)",
    }
    json.dump(man, open(f"{out}/manifest.json", 'w'), indent=0)

    def fmt(agg_d):
        return (f"MAE {np.mean(agg_d['mae']):.3f} RMSE {np.mean(agg_d['rmse']):.3f} "
                f"R2 {np.mean(agg_d['r2']):.3f}")
    print(f"[{dataset}] {name}: {n} graphs; target={target_name}; "
          f"mean {y.mean():.2f} std {y.std():.2f}; nq={NQ}")
    print(f"  -- baselines (raw units, mean over {len(SEEDS)} seed(s)) --")
    print(f"  {'K':>3} {'predict-mean':>30} {'Ridge(facts)':>30} {'GNN few-shot':>30}")
    for k in K_SHOTS:
        ks = str(k)
        mm = fmt(agg([m for m in meanb[ks]]))
        rr = fmt(agg([m for m in ridge[ks]]))
        gg = fmt(agg([m for m in gnn_fs[ks]])) if gnn_fs[ks] else "-"
        print(f"  {k:>3} {mm:>30} {rr:>30} {gg:>30}")
    if gnn_full:
        print(f"  GNN full-supervision (train {TRAIN_FULL}, K-independent upper bar): "
              f"{fmt(agg(gnn_full))}")
    print(f"  wrote {len(files)} prompt files + manifest -> {out}")
    return man


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'FreeSolv'
    os.makedirs(OUT_BASE, exist_ok=True)
    run(dataset)
    print("run run_qwen.py / run_opus_cli.py on each seed*_k*.txt; "
          "score with score_regression_icl.py")
