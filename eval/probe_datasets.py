"""Probe a broad set of real graph datasets across scientific domains (TUDatasets).
Downloads to /home/scratch/tudata and reports which load + size/classes/features,
so we can pick a wide cross-domain set for the sweep. Run in the fmsn venv."""
import sys, json
import numpy as np
from torch_geometric.datasets import TUDataset

ROOT = '/home/scratch/tudata'
# (name, domain) — broad coverage; some may fail/rename, that's fine (reported).
CAND = [
    ('MUTAG', 'chemistry'), ('PTC_MR', 'chemistry'), ('BZR', 'chemistry'),
    ('COX2', 'chemistry'), ('DHFR', 'chemistry'), ('AIDS', 'chemistry'),
    ('NCI1', 'chemistry'), ('Mutagenicity', 'chemistry'),
    ('PROTEINS', 'biology'), ('ENZYMES', 'biology'), ('DD', 'biology'),
    ('KKI', 'neuroscience'), ('OHSU', 'neuroscience'), ('Peking_1', 'neuroscience'),
    ('IMDB-BINARY', 'social'), ('IMDB-MULTI', 'social'), ('COLLAB', 'social'),
    ('REDDIT-BINARY', 'social'), ('deezer_ego_nets', 'social'),
    ('github_stargazers', 'social'), ('twitch_egos', 'social'),
    ('Letter-high', 'vision'), ('MSRC_21', 'vision'), ('Fingerprint', 'vision'),
    ('Synthie', 'synthetic'), ('COLORS-3', 'synthetic'), ('TRIANGLES', 'synthetic'),
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
json.dump([dict(zip(['name','domain','n','classes','avgN','avgE','has_x','ncat','chance'], r))
           for r in rows], open('/home/scratch/bench_out/probe_datasets.json', 'w'), indent=0)
print(f"\n{len(rows)} datasets OK across "
      f"{len(set(r[1] for r in rows))} domains -> /home/scratch/bench_out/probe_datasets.json")
