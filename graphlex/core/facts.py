"""facts(G) -> exact structural quantities, computed with NetworkX. NO LLM.

This dict is the single contract every downstream consumer reads (verbalize,
tabular/SGC serialization, agent skills). Computation is exact; presentation
lives elsewhere.
"""
from collections import Counter
import numpy as np
import networkx as nx


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


def _largest_cc(G):
    if G.number_of_nodes() == 0:
        return G
    return G.subgraph(max(nx.connected_components(G), key=len))


def _path_stats(G, n):
    """Avg shortest path length + diameter on the largest component. Exact for
    small graphs; skipped (None) above DIAMETER_NODE_CAP to stay cheap. No LLM."""
    DIAMETER_NODE_CAP = 1500
    if n < 2:
        return 0.0, 0
    Gc = _largest_cc(G)
    if Gc.number_of_nodes() < 2:
        return 0.0, 0
    if Gc.number_of_nodes() > DIAMETER_NODE_CAP:
        return None, None
    return float(nx.average_shortest_path_length(Gc)), int(nx.diameter(Gc))


def facts(G, node_attrs=None):
    node_attrs = list(node_attrs or [])
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = [d for _, d in G.degree()]
    mean_deg = float(np.mean(deg)) if deg else 0.0
    comms = _communities(G)
    bc = nx.betweenness_centrality(G) if n > 2 else {v: 0.0 for v in G}

    try:
        assort = float(nx.degree_assortativity_coefficient(G)) if m > 0 else float("nan")
    except Exception:
        assort = float("nan")
    if assort != assort:  # NaN -> not informative; keep as nan, rendered "undefined"
        pass
    apl, diam = _path_stats(G, n)

    out = {
        "structure": {
            "n_nodes": n,
            "n_edges": m,
            "density": nx.density(G),
            "n_components": nx.number_connected_components(G),
            "mean_degree": mean_deg,
            "max_degree": int(max(deg)) if deg else 0,
            "degree_std": float(np.std(deg)) if deg else 0.0,
            "max_over_mean_degree": (float(max(deg)) / mean_deg) if mean_deg > 0 else 0.0,
            "avg_clustering": float(nx.average_clustering(G)) if n else 0.0,
            "transitivity": float(nx.transitivity(G)) if n else 0.0,
            "degree_assortativity": assort,
            "avg_path_length": apl,   # largest component; None if too large
            "diameter": diam,         # largest component; None if too large
            "n_communities": len(comms),
        },
        "attributes": {},
        "_communities": comms,
    }

    for attr in node_attrs:
        vals = nx.get_node_attributes(G, attr)
        if not vals:
            continue
        groups = sorted({str(v) for v in vals.values()})
        try:
            assort = float(nx.attribute_assortativity_coefficient(G, attr))
        except Exception:
            assort = float("nan")
        cross_edges = [(u, v) for u, v in G.edges if vals.get(u) != vals.get(v)]
        by_group = {g: [bc[v] for v in G if str(vals.get(v)) == g] for g in groups}
        mean_bc = {g: (float(np.mean(by_group[g])) if by_group[g] else 0.0) for g in groups}
        out["attributes"][attr] = {
            "groups": groups,
            "assortativity": assort,
            "within_edges": m - len(cross_edges),
            "cross_edges": len(cross_edges),
            "cross_edge_list": cross_edges,
            "community_purity": _community_attr_purity(comms, vals),
            "mean_betweenness_by_group": mean_bc,
        }
    return out
