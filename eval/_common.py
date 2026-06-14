"""Shared helpers for the graphlex eval scorers.

Single source of truth for the pieces that were copy-pasted across the scoring
scripts, so a fix in one place fixes them all:

  * ``ANS_LINE`` / ``parse_ans`` -- the TOLERANT answer-line parser. Models (esp.
    Qwen) emit several near-format variants for the same line, e.g.::
        "0 CLASS1"   "Query 0 CLASS1"   "0: CLASS1"   "0) CLASS1"   "0 - CLASS1"
    All decode to {0: "CLASS1"}. The earlier strict ``^\\d+ TOKEN$`` regex
    silently dropped whole files that used the "Query N TOKEN" form (returning an
    empty parse -> the scorer skipped that seed), which under-counted seeds and
    produced wrong means. Always use this tolerant parser.
  * ``bal_acc`` -- balanced accuracy = macro-averaged per-class recall. On
    imbalanced test sets raw accuracy rewards always-predict-majority (which the
    LLM, seeing balanced shots, can't do); balanced accuracy makes majority ==
    chance and judges real per-class discrimination. This is the primary metric.
  * ``FKEYS`` / ``fvec`` / ``node_cats`` / ``to_nx`` / ``comp`` -- the graphlex
    facts() feature-vector extractor + node-category / composition helpers used to
    rebuild the classical (logreg) baseline at scoring time.

NOTE: this is BEHAVIOR-PRESERVING. ``FKEYS``/``fvec``/``node_cats``/``to_nx``/
``comp`` here are byte-for-byte the versions in ``sweep.py`` (the generator that
produced the sweep prompts + manifest), so importing them in the scorers yields
identical numbers.
"""
import re

import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx

# --- answer-line parsing -----------------------------------------------------
# tolerant: handles "0 CLASS0", "Query 0 CLASS0", "0: CLASS0", "0) CLASS0",
# "0 - CLASS0" (case-insensitive 'query'); token = letters/digits/underscore.
ANS_LINE = re.compile(r'^\s*(?:query\s*)?(\d+)\s*[:.\)\-]?\s+([A-Za-z0-9_]+)\s*$', re.I)


def parse_ans(path):
    """Parse an .ans file -> {int query id: UPPER token}."""
    d = {}
    with open(path) as fh:
        for ln in fh:
            m = ANS_LINE.match(ln.strip())
            if m:
                d[int(m.group(1))] = m.group(2).strip().upper()
    return d


# --- metrics -----------------------------------------------------------------
def bal_acc(truth_list, pred_map):
    """Balanced accuracy (macro-averaged per-class recall).

    truth_list: iterable of (query_id, label); pred_map: {query_id: TOKEN}.
    Labels are upper-cased for comparison. Returns None if truth is empty.
    """
    by = {}
    for i, lab in truth_list:
        by.setdefault(str(lab).upper(), []).append(i)
    recs = []
    for lab, ids in by.items():
        recs.append(np.mean([pred_map.get(i) == lab for i in ids]))
    return float(np.mean(recs)) if recs else None


def raw_acc(truth_list, pred_map):
    """Plain accuracy over the truth ids; None if truth empty or pred empty."""
    tt = {i: str(lab).upper() for i, lab in truth_list}
    if not tt or not pred_map:
        return None
    return sum(1 for i, lab in tt.items() if pred_map.get(i) == lab) / len(tt)


# --- graphlex facts() feature vector + node composition ----------------------
# (identical to sweep.py / label_curve.py / zero_label.py)
FKEYS = ['n_nodes', 'n_edges', 'density', 'n_components', 'mean_degree', 'max_degree',
         'degree_std', 'max_over_mean_degree', 'avg_clustering', 'transitivity',
         'degree_assortativity', 'avg_path_length', 'diameter', 'n_cycles', 'n_communities']


def fvec(f):
    """Fixed-length feature vector from a graphlex facts() dict (NaN/None -> 0.0)."""
    s = f['structure']
    return [0.0 if (s[k] is None or (isinstance(s[k], float) and s[k] != s[k])) else float(s[k])
            for k in FKEYS]


def node_cats(data):
    """Argmax category per node for clean one-hot categorical node features.

    Returns (cats array, n_categories) or (None, 0) if x is absent / not one-hot.
    """
    x = data.x
    if x is None:
        return None, 0
    xs = x.numpy()
    if not (np.allclose(xs.sum(1), 1) and set(np.unique(xs).tolist()) <= {0.0, 1.0}):
        return None, 0
    return xs.argmax(1), xs.shape[1]


def to_nx(data, cats):
    """PyG Data -> undirected networkx graph (self-loops removed), optionally
    annotating each node with a 'type' attribute t<cat>."""
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    if cats is not None:
        nx.set_node_attributes(G, {i: f"t{int(cats[i])}" for i in G.nodes()}, 'type')
    return G


def comp(cats, ncat):
    """Fixed-length node-type composition (category fractions); [] if no cats."""
    if cats is None or ncat == 0:
        return []
    v = np.bincount(cats, minlength=ncat).astype(float)
    return (v / v.sum()).tolist() if v.sum() else v.tolist()
