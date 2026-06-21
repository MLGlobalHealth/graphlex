"""facts(G) -> exact structural quantities, computed with NetworkX. NO LLM.

This dict is the single contract every downstream consumer reads (verbalize, the
logreg feature vector, agent skills, the web demo). Computation is exact;
presentation lives in verbalize. The canonical feature set is groups A-K
(see GROUPS); both verbalize() and feature_vector() select from it so experiments
can ablate by group.
"""
from collections import Counter
import numpy as np
import networkx as nx

# Bump when the feature set / definitions change; record in result manifests so
# runs on different feature versions are never conflated.
FEATURE_VERSION = "A-K/v1"

# Graph-level SCALAR feature groups (for the logreg vector + ablations). Groups I
# (named top-k) and K (per-node detail) are node-level -> rendered by verbalize(),
# not part of the flat scalar vector.
GROUPS = {
    "A": ["n_nodes", "n_edges", "density", "n_components"],
    "B": ["mean_degree", "max_degree", "degree_std", "max_over_mean_degree",
          "degree_skewness", "degree_kurtosis", "degree_gini", "powerlaw_alpha"],
    "C": ["avg_clustering", "transitivity", "n_triangles", "n_squares"],
    "D": ["avg_path_length", "diameter", "radius"],
    "E": ["degree_assortativity", "n_communities", "modularity"],
    "F": ["max_kcore", "spectral_gap"],
    "G": ["n_cycles"],
    "H": ["betweenness_mean", "betweenness_max", "closeness_mean",
          "eigenvector_mean", "eigenvector_max"],
    "J": ["er_clustering_ratio", "config_clustering", "config_clustering_ratio",
          "smallworld_sigma", "smallworld_omega"],
}
SCALAR_GROUPS = "ABCDEFGHJ"   # groups that appear in feature_vector()
ALL_GROUPS = "ABCDEFGHIJK"


def feature_names(groups=SCALAR_GROUPS):
    """Ordered scalar feature names for the selected groups (the logreg columns)."""
    return [k for g in groups for k in GROUPS.get(g, [])]


def feature_vector(f, groups=SCALAR_GROUPS):
    """Flat scalar vector from a facts() dict for the selected groups (NaN/None->0.0).
    The single canonical logreg feature builder — eval scripts import this."""
    s = f["structure"]
    out = []
    for k in feature_names(groups):
        v = s.get(k)
        out.append(0.0 if (v is None or (isinstance(v, float) and v != v)) else float(v))
    return out


def _gini(x):
    x = np.sort(np.asarray(x, float)); nn = len(x)
    if nn == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, nn + 1)) * x) / (nn * x.sum())) - (nn + 1) / nn)


def _communities(G):
    """Deterministic community detection (greedy modularity — no RNG)."""
    if G.number_of_edges() == 0:
        return [{v} for v in G.nodes()]
    return [set(c) for c in nx.community.greedy_modularity_communities(G)]


def _community_attr_purity(comms, vals):
    """Mean fraction of nodes whose attribute == their community's majority."""
    total = matched = 0
    for c in comms:
        labs = [vals[v] for v in c if v in vals]
        if not labs:
            continue
        matched += Counter(labs).most_common(1)[0][1]
        total += len(labs)
    return (matched / total) if total else None


def _eig_centrality(G):
    """Eigenvector centrality, DETERMINISTIC. nx.eigenvector_centrality_numpy uses
    ARPACK with a random start vector (top-k ties flip run-to-run); for n<=2000 we
    use a dense symmetric eigendecomposition instead (reproducible), matching the
    networkx convention (unit L2 norm, positive orientation). ARPACK fallback above."""
    n = G.number_of_nodes()
    if n == 0 or G.number_of_edges() == 0:
        return {v: 0.0 for v in G}
    if n <= 2000:
        A = nx.to_numpy_array(G, nodelist=list(G))
        _, V = np.linalg.eigh(A)
        x = V[:, -1]                       # principal eigenvector (largest eigenvalue)
        if x.sum() < 0:
            x = -x                         # positive orientation
        return {v: float(x[i]) for i, v in enumerate(G)}
    try:
        return {v: float(x) for v, x in nx.eigenvector_centrality_numpy(G).items()}
    except Exception:
        return {v: 0.0 for v in G}


def _largest_cc(G):
    if G.number_of_nodes() == 0:
        return G
    return G.subgraph(max(nx.connected_components(G), key=len))


def _ring_sizes(G, n, m):
    """Histogram of ring sizes from a minimum cycle basis. None above caps."""
    if n < 3 or m < n:
        return {}
    if n > 400 or m > 1200 or (m - n) > 60:
        return None
    try:
        basis = nx.minimum_cycle_basis(G)
    except Exception:
        return None
    h = {}
    for cyc in basis:
        h[len(cyc)] = h.get(len(cyc), 0) + 1
    return dict(sorted(h.items()))


def facts(G, node_attrs=None, nodes=True):
    """Exact structural quantities (groups A-K). `nodes=True` adds the node-level
    section (I named top-k + K per-node detail with neighbor lists); the graph-level
    scalar set (A-H,J) is always computed. NO LLM."""
    node_attrs = list(node_attrs or [])
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg_c = dict(G.degree())
    dega = np.array([deg_c[v] for v in G], float) if n else np.array([])
    mean_deg = float(dega.mean()) if n else 0.0
    dens = float(nx.density(G))
    comms = _communities(G)
    Gc = _largest_cc(G)
    DIAMETER_NODE_CAP = 1500

    # Centralities feed groups H/J, the node section, and structure x attributes.
    # Size-gated, fixed-seed betweenness for large graphs; graceful fallbacks.
    if n > 2:
        bc = (nx.betweenness_centrality(G) if n <= 400
              else nx.betweenness_centrality(G, k=min(100, n), seed=0))
    else:
        bc = {v: 0.0 for v in G}
    eig = _eig_centrality(G)
    clo = nx.closeness_centrality(G) if n else {}

    # B: degree-distribution shape
    sd = float(dega.std()) if n else 0.0
    skew = float(np.mean(((dega - mean_deg) / sd) ** 3)) if sd > 0 else 0.0
    kurt = float(np.mean(((dega - mean_deg) / sd) ** 4) - 3) if sd > 0 else 0.0
    alpha = float(1 + n / np.sum(np.log(dega / 0.5))) if (n and np.all(dega > 0)) else float("nan")

    # C: triangles, squares (4-cycles)
    n_tri = sum(nx.triangles(G).values()) // 3 if n else 0
    if n and n <= 400:
        A = nx.to_numpy_array(G)
        n_sq = int(round((np.trace(np.linalg.matrix_power(A, 4)) - 2 * np.sum(dega ** 2) + 2 * m) / 8))
    else:
        n_sq = None

    # D: paths on the largest component
    if Gc.number_of_nodes() > 1 and Gc.number_of_nodes() <= DIAMETER_NODE_CAP:
        apl = float(nx.average_shortest_path_length(Gc)); diam = int(nx.diameter(Gc)); rad = int(nx.radius(Gc))
    elif Gc.number_of_nodes() > 1:
        apl = diam = rad = None
    else:
        apl, diam, rad = 0.0, 0, 0

    # E: assortativity, modularity
    try:
        assort = float(nx.degree_assortativity_coefficient(G)) if m > 0 else float("nan")
    except Exception:
        assort = float("nan")
    try:
        modularity = float(nx.community.modularity(G, comms)) if m else 0.0
    except Exception:
        modularity = float("nan")

    # F: cores, spectral
    max_kcore = int(max(nx.core_number(G).values())) if m else 0
    try:
        spectral_gap = float(nx.algebraic_connectivity(Gc)) if Gc.number_of_nodes() > 1 else 0.0
    except Exception:
        spectral_gap = float("nan")

    # H: centrality scalar summaries
    bmean = float(np.mean(list(bc.values()))) if n else 0.0
    bmax = float(max(bc.values())) if n else 0.0
    cmean = float(np.mean(list(clo.values()))) if n else 0.0
    emean = float(np.mean(list(eig.values()))) if n else 0.0
    emax = float(max(eig.values())) if n else 0.0

    # J: null-model contrasts (analytic where possible; see FEATURE_SYNC_CHECKLIST)
    ac = float(nx.average_clustering(G)) if n else 0.0
    k2 = float((dega ** 2).mean()) if n else 0.0
    config_clustering = ((k2 - mean_deg) ** 2 / (n * mean_deg ** 3)) if (n and mean_deg > 0) else 0.0
    er_ratio = (ac / dens) if dens > 0 else float("nan")
    config_ratio = (ac / config_clustering) if config_clustering > 0 else float("nan")
    Lr = (np.log(n) / np.log(mean_deg)) if mean_deg > 1 else float("nan")
    L = apl if (apl not in (None, 0.0)) else float("nan")
    Cl = (3 * (mean_deg - 2) / (4 * (mean_deg - 1))) if mean_deg > 1 else float("nan")
    sigma = ((ac / dens) / (L / Lr)) if (dens > 0 and L == L and Lr == Lr and L) else float("nan")
    omega = ((Lr / L) - (ac / Cl)) if (L == L and L and Cl == Cl and Cl) else float("nan")

    nodes_sec = None
    if nodes:
        node2c = {v: i for i, c in enumerate(comms) for v in c}

        def _topk(d, k=5):
            # secondary key str(node) -> deterministic order even on exact value ties
            return sorted(d, key=lambda x: (-d[x], str(x)))[:k]

        per_node = None
        if n <= 200:   # K: per-node detail + neighbor lists (small graphs only)
            per_node = {str(v): {"degree": deg_c[v], "betweenness": round(bc[v], 3),
                                 "eigenvector": round(eig[v], 3), "closeness": round(clo.get(v, 0.0), 3),
                                 "community": node2c.get(v, 0),
                                 "neighbors": [str(u) for u in G.neighbors(v)]} for v in G}
        nodes_sec = {
            "names": list(G.nodes()),
            "degree": deg_c, "betweenness": bc, "eigenvector": eig, "closeness": clo,
            "community": node2c,
            "top_degree": _topk(deg_c), "top_betweenness": _topk(bc), "top_eigenvector": _topk(eig),
            "min_degree_node": (min(deg_c, key=lambda x: deg_c[x]) if deg_c else None),
            "n_isolates": int(sum(1 for x in deg_c.values() if x == 0)),
            "per_node": per_node,
        }

    out = {
        "feature_version": FEATURE_VERSION,
        "structure": {
            "n_nodes": n, "n_edges": m, "density": dens,
            "n_components": nx.number_connected_components(G),
            "mean_degree": mean_deg, "max_degree": int(dega.max()) if n else 0,
            "degree_std": sd, "max_over_mean_degree": (float(dega.max()) / mean_deg) if mean_deg > 0 else 0.0,
            "degree_skewness": skew, "degree_kurtosis": kurt, "degree_gini": _gini(dega),
            "powerlaw_alpha": alpha,
            "avg_clustering": ac, "transitivity": float(nx.transitivity(G)) if n else 0.0,
            "n_triangles": int(n_tri), "n_squares": n_sq,
            "avg_path_length": apl, "diameter": diam, "radius": rad,
            "degree_assortativity": assort, "n_communities": len(comms), "modularity": modularity,
            "max_kcore": max_kcore, "spectral_gap": spectral_gap,
            "n_cycles": m - n + nx.number_connected_components(G) if n else 0,
            "ring_sizes": _ring_sizes(G, n, m),
            "betweenness_mean": bmean, "betweenness_max": bmax, "closeness_mean": cmean,
            "eigenvector_mean": emean, "eigenvector_max": emax,
            "er_clustering_ratio": er_ratio, "config_clustering": config_clustering,
            "config_clustering_ratio": config_ratio,
            "smallworld_sigma": sigma, "smallworld_omega": omega,
        },
        "attributes": {},
        "nodes": nodes_sec,
        "_communities": comms,
    }

    for attr in node_attrs:
        vals = nx.get_node_attributes(G, attr)
        if not vals:
            continue
        groups = sorted({str(v) for v in vals.values()})
        comp_counts = Counter(str(v) for v in vals.values())
        total_attr = sum(comp_counts.values())
        composition = {g: comp_counts[g] / total_attr for g in groups} if total_attr else {}
        try:
            a_assort = float(nx.attribute_assortativity_coefficient(G, attr))
        except Exception:
            a_assort = float("nan")
        et_counts = Counter()
        for u, v in G.edges:
            if u in vals and v in vals:
                a, b = sorted((str(vals[u]), str(vals[v])))
                et_counts[f"{a}-{b}"] += 1
        et_total = sum(et_counts.values())
        edge_type_composition = ({k: c / et_total for k, c in et_counts.items()} if et_total else {})
        cross_edges = [(u, v) for u, v in G.edges if vals.get(u) != vals.get(v)]
        by_group = {g: [bc[v] for v in G if str(vals.get(v)) == g] for g in groups}
        mean_bc = {g: (float(np.mean(by_group[g])) if by_group[g] else 0.0) for g in groups}
        out["attributes"][attr] = {
            "groups": groups, "composition": composition,
            "edge_type_composition": edge_type_composition, "assortativity": a_assort,
            "within_edges": m - len(cross_edges), "cross_edges": len(cross_edges),
            "cross_edge_list": cross_edges, "community_purity": _community_attr_purity(comms, vals),
            "mean_betweenness_by_group": mean_bc,
        }
    return out
