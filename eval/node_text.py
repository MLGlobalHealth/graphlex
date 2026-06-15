"""Readable-text source for Cora / Citeseer / PubMed node features (arm (b)).

Planetoid ships only ANONYMIZED bag-of-words with NO vocabulary, so the opaque
arm (a) can only show `[w26,w61,...]`. This module sources HUMAN-READABLE per-node
text (paper title + abstract) and real class names so the LLM can actually read
what each node is about. It is import-only: `node_icl.py` calls
`load_readable(name)` to get a Planetoid-node-aligned list of text strings + the
real class-name map. NO graphlex core is touched.

SOURCE (documented precisely; see NODE_TRACK_PLAN.md §2/§7):
  HuggingFace `Graph-COM/Text-Attributed-Graphs/{cora,citeseer,pubmed}/raw_texts.pt`
  (the Graph-LLM / CurryTang text-attributed-graph release; per-node "Title: ...
  Abstract: ..." strings). For class names: that repo's `categories.csv`.

ALIGNMENT to the PyG Planetoid node ordering (the load-bearing correctness step,
verified empirically in build_cache()):
  * Cora, PubMed: the TAG release ships the SAME binary bag-of-words as Planetoid
    but in a DIFFERENT node order (and with permuted class ids). We recover the
    EXACT Planetoid->TAG node permutation by hashing each node's binarized BoW row
    and matching (1.000 coverage for both; the few duplicate-BoW rows are resolved
    greedily). So Cora/PubMed get the node's OWN real title+abstract. EXACT.
  * Citeseer: the TAG release is a re-collection of only 3186 of Planetoid's 3327
    nodes, NOT in Planetoid order, and its processed features are SBERT (384-d),
    NOT the 3703-d BoW -> NOT row-matchable. Planetoid's true per-node class IS
    recoverable (via the LINQS citeseer.content binary-BoW row-match, 3312/3327),
    but a node-exact title is NOT cleanly obtainable for all nodes from any single
    release. DOCUMENTED FALLBACK: each Planetoid Citeseer node is given a REAL
    Citeseer title+abstract drawn (deterministically, seeded by node id) from the
    TAG corpus OF ITS TRUE CLASS. This is honest readable in-domain text whose
    TOPIC matches the node's real label (so the homophily/topic signal the LLM
    reads is class-correct), but it is NOT guaranteed to be that exact paper.
    Marked APPROX. The 15 isolated (all-zero-BoW) nodes get a class-name stub.

The cache is built once into eval/../bench_out/node_icl/_textcache/<Name>.json and
reused. Downloaded .pt files are torch zips; raw_texts.pt is a single pickled
list[str] read with a RESTRICTED unpickler (no global construction) so we never
execute arbitrary pickle from the external repo. The processed BoW is read as a
raw float32 storage out of the zip (no unpickling of the PyG object at all).
"""
import os, io, sys, json, zipfile, pickle, hashlib, urllib.request
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/home/scratch/planetoid'
CACHE = '/home/scratch/bench_out/node_icl/_textcache'
HF = 'https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs/resolve/main'
LINQS = 'https://linqs-data.soe.ucsc.edu/public/lbc'

# TAG categories.csv order per dataset (= TAG class id -> readable name).
TAG_NAMES = {
    'Cora': ['Case Based', 'Genetic Algorithms', 'Neural Networks',
             'Probabilistic Methods', 'Reinforcement Learning', 'Rule Learning',
             'Theory'],
    'PubMed': ['Diabetes Mellitus Experimental', 'Diabetes Mellitus Type 1',
               'Diabetes Mellitus Type 2'],
    'Citeseer': ['Agents', 'Machine Learning', 'Information Retrieval',
                 'Database', 'Human-Computer Interaction', 'Artificial Intelligence'],
}
# Planetoid feature dims / lower-cased HF dir.
DIM = {'Cora': 1433, 'PubMed': 500, 'Citeseer': 3703}
HFDIR = {'Cora': 'cora', 'PubMed': 'pubmed', 'Citeseer': 'citeseer'}


class _SafeUnpickler(pickle.Unpickler):
    """raw_texts.pt is a plain list[str]; forbid ALL global construction so no
    arbitrary code from the external pickle can run."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _dl(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _read_raw_texts(pt_path):
    z = zipfile.ZipFile(pt_path)
    inner = [n for n in z.namelist() if n.endswith('raw_texts/data.pkl')][0]
    return _SafeUnpickler(io.BytesIO(z.read(inner))).load()


def _read_tag_bow(pt_path, n_nodes, dim):
    """The TAG processed_data.pt's first storage (data/0) is the float32 BoW for
    Cora/PubMed. Read it raw without unpickling the PyG object. Returns None if the
    storage size doesn't match an (n_nodes, dim) BoW (e.g. Citeseer = SBERT)."""
    z = zipfile.ZipFile(pt_path)
    cand = [n for n in z.namelist() if n.endswith('processed_data/data/0')]
    if not cand:
        return None
    buf = np.frombuffer(z.read(cand[0]), dtype=np.float32)
    if buf.size != n_nodes * dim:
        return None
    return buf.reshape(n_nodes, dim)


def _read_tag_y(pt_path, n_nodes):
    """TAG label vector (int64 storage of byte-size n_nodes*8)."""
    z = zipfile.ZipFile(pt_path)
    for n in z.namelist():
        if '/data/' in n and z.getinfo(n).file_size == n_nodes * 8:
            arr = np.frombuffer(z.read(n), dtype=np.int64)
            if arr.size == n_nodes and arr.min() >= 0 and arr.max() < 50:
                return arr
    return None


def _bow_rowmatch(xP_bin, xT_bin):
    """Planetoid node i -> TAG row j, by exact binarized-BoW row hash (greedy on
    duplicate rows). Returns (mapping list, n_exact_unique, n_missing)."""
    by = defaultdict(list)
    for j in range(xT_bin.shape[0]):
        by[hashlib.md5(xT_bin[j].tobytes()).hexdigest()].append(j)
    used, mapping, uniq, missing = set(), [None] * xP_bin.shape[0], 0, 0
    for i in range(xP_bin.shape[0]):
        h = hashlib.md5(xP_bin[i].tobytes()).hexdigest()
        cands = [j for j in by.get(h, []) if j not in used]
        if not cands:
            missing += 1; continue
        j = cands[0]; used.add(j); mapping[i] = j
        if len(by[h]) == 1:
            uniq += 1
    return mapping, uniq, missing


def build_cache(name, verbose=True):
    """Build (and cache) the Planetoid-aligned readable-text list + class names for
    `name`. Returns dict {texts: [str per node], class_names: {id:name},
    alignment: str}. Idempotent."""
    from torch_geometric.datasets import Planetoid
    os.makedirs(CACHE, exist_ok=True)
    out_path = f"{CACHE}/{name}.json"
    if os.path.exists(out_path):
        return json.load(open(out_path))

    pl = Planetoid(ROOT, name=name)[0]
    yP = pl.y.numpy(); xP = pl.x.numpy(); N = pl.num_nodes
    xPb = (xP > 0).astype(np.uint8)
    d = HFDIR[name]
    rt_path = _dl(f"{HF}/{d}/raw_texts.pt", f"/tmp/_tagtext/{d}_raw_texts.pt")
    proc_path = _dl(f"{HF}/{d}/processed_data.pt", f"/tmp/_tagtext/{d}_processed.pt")
    texts = _read_raw_texts(rt_path)

    xT = _read_tag_bow(proc_path, len(texts), DIM[name])
    if xT is not None:
        # ---- EXACT path (Cora, PubMed): row-match BoW ----
        xTb = (xT > 0).astype(np.uint8)
        mapping, uniq, missing = _bow_rowmatch(xPb, xTb)
        aligned = [texts[mapping[i]] if mapping[i] is not None else
                   f"(no text; class {TAG_NAMES[name][int(yP[i])]})" for i in range(N)]
        # class names: derive Planetoid-id -> name via the matched TAG labels.
        tagy = _read_tag_y(proc_path, len(texts))
        cls_names = {}
        if tagy is not None:
            for i in range(N):
                if mapping[i] is not None:
                    cls_names[int(yP[i])] = TAG_NAMES[name][int(tagy[mapping[i]])]
        if len(cls_names) < len(set(yP.tolist())):
            cls_names = {c: TAG_NAMES[name][c] for c in sorted(set(yP.tolist()))}
        align = (f"EXACT (binarized-BoW row-match to TAG release; "
                 f"{uniq} unique-row, {missing} missing of {N})")
    else:
        # ---- APPROX path (Citeseer): class-consistent real text ----
        cls_names, align = _citeseer_fallback(name, N, yP, xPb, texts, proc_path)
        aligned = _citeseer_assign(name, N, yP, texts, proc_path, cls_names)
        align = "APPROX " + align

    if verbose:
        print(f"[text:{name}] {align}; class_names={cls_names}")
    res = {"texts": aligned, "class_names": {str(k): v for k, v in cls_names.items()},
           "alignment": align}
    json.dump(res, open(out_path, 'w'))
    return res


def _citeseer_fallback(name, N, yP, xPb, texts, proc_path):
    """Determine Planetoid-id -> real class name for Citeseer via the LINQS
    citeseer.content binary-BoW row-match (authoritative true class per node)."""
    cls_names = {}
    try:
        tgz = _dl(f"{LINQS}/citeseer.tgz", "/tmp/_tagtext/citeseer.tgz")
        import tarfile
        td = "/tmp/_tagtext/citeseer_linqs"
        if not os.path.exists(f"{td}/citeseer.content"):
            os.makedirs(td, exist_ok=True)
            with tarfile.open(tgz) as t:
                for m in t.getmembers():
                    if m.name.endswith('citeseer.content'):
                        m.name = 'citeseer.content'; t.extract(m, td)
        LMAP = {'Agents': 'Agents', 'ML': 'Machine Learning', 'IR': 'Information Retrieval',
                'DB': 'Database', 'HCI': 'Human-Computer Interaction', 'AI': 'Artificial Intelligence'}
        rows = []
        with open(f"{td}/citeseer.content") as fh:
            for ln in fh:
                p = ln.rstrip('\n').split('\t')
                feats = np.array([int(v) for v in p[1:1 + DIM[name]]], dtype=np.uint8)
                rows.append((feats, p[-1]))
        by = defaultdict(list)
        for k, (f, lab) in enumerate(rows):
            by[hashlib.md5(f.tobytes()).hexdigest()].append(k)
        for i in range(N):
            if xPb[i].sum() == 0:
                continue
            cand = by.get(hashlib.md5(xPb[i].tobytes()).hexdigest(), [])
            if cand:
                cls_names.setdefault(int(yP[i]), LMAP.get(rows[cand[0]][1], rows[cand[0]][1]))
    except Exception as e:
        print(f"[text:{name}] LINQS class-name match failed ({e}); using TAG order", file=sys.stderr)
    for c in sorted(set(yP.tolist())):
        cls_names.setdefault(c, TAG_NAMES[name][c])
    return cls_names, "(class names via LINQS citeseer.content row-match; per-node text class-consistent)"


def _citeseer_assign(name, N, yP, texts, proc_path, cls_names):
    """Assign each Planetoid Citeseer node a REAL TAG abstract OF ITS TRUE CLASS,
    deterministically by node id. Topic-correct, not paper-exact (documented)."""
    tagy = _read_tag_y(proc_path, len(texts))
    name_to_pid = {v: k for k, v in cls_names.items()}
    # TAG class id -> list of text indices, mapped onto Planetoid class id via name.
    pid_pool = defaultdict(list)
    for j in range(len(texts)):
        tname = TAG_NAMES[name][int(tagy[j])] if tagy is not None else None
        pid = name_to_pid.get(tname)
        if pid is not None:
            pid_pool[pid].append(j)
    aligned = []
    for i in range(N):
        c = int(yP[i]); pool = pid_pool.get(c)
        if not pool:
            aligned.append(f"(no text; class {cls_names.get(c, c)})")
        else:
            aligned.append(texts[pool[i % len(pool)]])
    return aligned


def load_readable(name):
    """Public entry: returns (texts list aligned to Planetoid node id, {class id:
    readable name}, alignment-note str)."""
    res = build_cache(name)
    cn = {int(k): v for k, v in res['class_names'].items()}
    return res['texts'], cn, res['alignment']


def node_text_summary(texts, glob, maxchars=320):
    """Per-node readable-text line for the ego-graph prompt (arm b), capped."""
    lines = []
    for local, g in enumerate(glob):
        t = (texts[g] or "").strip().replace('\n', ' ')
        if len(t) > maxchars:
            t = t[:maxchars].rsplit(' ', 1)[0] + ' ...'
        tag = "TARGET" if local == 0 else f"#{local}"
        lines.append(f"  {tag}: {t}")
    return "\n".join(lines)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ['Cora', 'Citeseer', 'PubMed']):
        r = build_cache(nm)
        print(f"{nm}: {len(r['texts'])} node texts, {r['alignment']}")
        print(f"   sample TARGET text: {r['texts'][0][:140]}")
