"""Probe a broad set of real graph datasets across scientific domains (TUDatasets).
Downloads to /home/scratch/tudata and reports which load + size/classes/features,
so we can pick a wide cross-domain set for the sweep. Run in the fmsn venv."""
import sys, json
import numpy as np
from torch_geometric.datasets import TUDataset

ROOT = '/home/scratch/tudata'
MERGE = '/home/scratch/bench_out/probe_datasets.json'  # append to existing if present
# (name, domain) — NEW domains/datasets to add to the sweep. Some may fail/rename.
CAND = [
    ('DBLP_v1', 'citation'),            # scholarly citation/coauthor ego-graphs
    ('FIRSTMM_DB', 'robotics'),         # 3D point-cloud object graphs (robotics)
    ('Cuneiform', 'archaeology'),       # cuneiform sign graphs (epigraphy)
    ('COIL-DEL', 'vision'),             # object images -> graphs
    ('FRANKENSTEIN', 'chemistry'),      # molecules (MNIST-substituted atoms)
    ('REDDIT-MULTI-5K', 'social'),      # online discussion threads (5 classes)
    ('MSRC_9', 'vision'),               # semantic image graphs
    ('SYNTHETICnew', 'synthetic'),
]

rows = []
for name, dom in CAND:
    try:
        ds = TUDataset(ROOT, name=name)
        ng = len(ds)
        nc = ds.num_classes
        sample = ds[:min(100, ng)]
        avg_n = float(np.mean([d.num_nodes for d in sample]))
        avg_m = float(np.mean([d.edge_index.size(1) // 2 for d in sample]))
        has_x = ds[0].x is not None
        ncat = ds[0].x.shape[1] if has_x else 0
        # class balance (chance = majority fraction)
        ys = np.array([int(ds[i].y) for i in range(ng)])
        chance = float(np.bincount(ys).max() / ng)
        rows.append((name, dom, ng, nc, round(avg_n, 1), round(avg_m, 1), has_x, ncat, round(chance, 3)))
        print(f"OK  {name:18} {dom:12} n={ng:5} cls={nc} avgN={avg_n:6.1f} avgE={avg_m:6.1f} "
              f"feat={has_x}({ncat}) chance={chance:.3f}", flush=True)
    except Exception as e:
        print(f"FAIL {name:18} {dom:12} {type(e).__name__}: {str(e)[:90]}", flush=True)
new = [dict(zip(['name', 'domain', 'n', 'classes', 'avgN', 'avgE', 'has_x', 'ncat', 'chance'], r))
       for r in rows]
import os
existing = json.load(open(MERGE)) if os.path.exists(MERGE) else []
have = {d['name'] for d in existing}
merged = existing + [d for d in new if d['name'] not in have]
json.dump(merged, open(MERGE, 'w'), indent=0)
print(f"\n{len(new)} new OK; merged total {len(merged)} datasets across "
      f"{len(set(d['domain'] for d in merged))} domains -> {MERGE}")
