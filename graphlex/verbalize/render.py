"""verbalize(facts) -> deterministic prose. Templates + threshold tables, NO LLM.

Every sentence is a fixed template filled from exact numbers in `facts`; the only
non-numeric choices come from graphlex.thresholds (versioned, inspectable).

The structural prose is organized by the canonical feature groups A-K (see
graphlex.core.facts.GROUPS): graph-level scalars (A-H, J) plus the node-level
groups I (named top-k) and K (per-node detail). `verbalize(..., groups=...)`
selects a subset so experiments can ablate by group; the rendered text and the
logreg feature_vector() then carry the identical feature set.
"""
import networkx as nx
from ..core.facts import ALL_GROUPS
from ..thresholds import homophily_word, correspondence_word, CENTRALITY_RATIO

STRUCTURE_GROUPS = "ABCDEFGHJ"   # graph-level scalar groups
NODE_GROUPS = "IK"               # node-level groups (need facts(nodes=True))


def _fmt(groups):
    return ", ".join(str(g) for g in groups)


def _p(n, noun):
    """'1 node' / '2 nodes' — correct singular/plural."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _g(x, fmt="{:.2f}", na="n/a"):
    """Format a possibly-None/NaN scalar."""
    if x is None or (isinstance(x, float) and x != x):
        return na
    return fmt.format(x)


def _topk_named(ids, valmap, fmt="{:.3g}"):
    return ", ".join(f"{v} ({fmt.format(valmap[v])})" for v in ids)


# --- per-group structural renderers (read exact numbers from facts) -----------

def _group_line(letter, f):
    """One sentence for group `letter`, or '' if not applicable to this graph."""
    s = f["structure"]
    n, m = s["n_nodes"], s["n_edges"]

    if letter == "A":
        return (f"Size & connectivity: {n} nodes, {m} edges, density "
                f"{s['density']:.2f}, {s['n_components']} connected component(s).")
    if letter == "B" and n:
        return (f"Degree distribution: mean {s['mean_degree']:.1f}, max "
                f"{s['max_degree']}, std {s['degree_std']:.1f}, max/mean "
                f"{s['max_over_mean_degree']:.1f}; skewness {s['degree_skewness']:.2f}, "
                f"kurtosis {s['degree_kurtosis']:.2f}, Gini {s['degree_gini']:.2f}; "
                f"rough power-law exponent ~{_g(s['powerlaw_alpha'], '{:.1f}')} "
                f"(x_min=1; unreliable for small n).")
    if letter == "C" and n:
        nsq = s.get("n_squares")
        sq = f"{nsq} 4-cycles" if nsq is not None else "4-cycle count skipped (large)"
        return (f"Clustering & cohesion: avg clustering {s['avg_clustering']:.2f}, "
                f"transitivity {s['transitivity']:.2f}; {s['n_triangles']} triangles, {sq}.")
    if letter == "D" and s.get("avg_path_length") is not None and s["avg_path_length"]:
        return (f"Distances (largest component): avg path length "
                f"{s['avg_path_length']:.2f}, diameter {s['diameter']}, "
                f"radius {s['radius']}.")
    if letter == "E" and m:
        return (f"Mixing & community: degree assortativity "
                f"{_g(s['degree_assortativity'], '{:+.2f}')}; {s['n_communities']} "
                f"communities (modularity Q = {_g(s['modularity'])}).")
    if letter == "F" and n:
        return (f"Cores & spectral: degeneracy (max k-core) {s['max_kcore']}; "
                f"algebraic connectivity {_g(s['spectral_gap'], '{:.3f}')}.")
    if letter == "G" and n:
        return f"Cycles: {s['n_cycles']} independent cycles (cyclomatic number)."
    if letter == "H" and n > 2:
        return (f"Centrality (graph summaries): betweenness mean "
                f"{s['betweenness_mean']:.3f} / max {s['betweenness_max']:.3f}; "
                f"closeness mean {s['closeness_mean']:.2f}; eigenvector mean "
                f"{s['eigenvector_mean']:.2f} / max {s['eigenvector_max']:.2f}.")
    if letter == "J" and m and n > 2:
        return (f"Null-model contrasts: clustering {s['avg_clustering']:.2f} vs ER "
                f"{s['density']:.2f} ({_g(s['er_clustering_ratio'], '{:.1f}')}x) and "
                f"configuration model {s['config_clustering']:.2f} "
                f"({_g(s['config_clustering_ratio'], '{:.1f}')}x); small-world sigma "
                f"{_g(s['smallworld_sigma'])}, omega {_g(s['smallworld_omega'])}. "
                f"(small n -> wide.)")
    return ""


def _node_group_line(letter, f):
    """Node-level group I (key nodes) / K (per-node detail). '' if unavailable."""
    nd = f.get("nodes")
    if not nd or not nd.get("degree"):
        return ""
    deg, bc, eig = nd["degree"], nd["betweenness"], nd["eigenvector"]

    if letter == "I":
        parts = ["most connected " + _topk_named(nd["top_degree"], deg)]
        if any(x > 0 for x in bc.values()):
            parts.append("top brokers (betweenness) " + _topk_named(nd["top_betweenness"], bc))
        if any(x > 0 for x in eig.values()):
            parts.append("most central (eigenvector) " + _topk_named(nd["top_eigenvector"], eig))
        return "Key nodes: " + "; ".join(parts) + "."

    if letter == "K":
        per = nd.get("per_node")
        if not per:
            n = f["structure"]["n_nodes"]
            return (f"Per-node detail omitted (n={n} too large); "
                    f"query a specific node's ego-graph instead.")
        lines = [
            f"  {v}: degree {d['degree']}, betweenness {d['betweenness']:.2f}, "
            f"eigenvector {d['eigenvector']:.2f}, community {d['community']}; "
            f"neighbors: {', '.join(d['neighbors'])}"
            for v, d in per.items()
        ]
        return ("Per-node detail (label: degree, betweenness, eigenvector, "
                "community; neighbors):\n" + "\n".join(lines))
    return ""


# --- attributes (separate from the A-K structural set) ------------------------

def _composition_phrase(comp, attr, k=8):
    """Top-k group fractions, descending then alphabetical; rest folded into 'other'."""
    items = sorted(comp.items(), key=lambda kv: (-kv[1], kv[0]))
    head = items[:k]
    shown = ", ".join(f"{g} {p*100:.0f}%" for g, p in head)
    rest = sum(p for _, p in items[k:])
    if rest > 0:
        shown += f", other {rest*100:.0f}%"
    return f"Node composition by {attr} ({len(comp)} groups): {shown}."


def _attribute(f, attr, a):
    sent = []
    if a.get("composition"):
        sent.append(_composition_phrase(a["composition"], attr))
    if a.get("edge_type_composition"):
        items = sorted(a["edge_type_composition"].items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        bonds = ", ".join(f"{k} {p*100:.0f}%" for k, p in items)
        sent.append(f"Edges by {attr}-pair: {bonds}.")
    sent.append(f"Ties are {homophily_word(a['assortativity'])} on {attr} "
                f"(assortativity {a['assortativity']:.2f}).")

    ncomm = f["structure"]["n_communities"]
    if a["community_purity"] is not None and ncomm > 1:
        sent.append(
            f"The {ncomm} detected communities correspond "
            f"{correspondence_word(a['community_purity'])} to the {attr} groups "
            f"({_fmt(a['groups'])})."
        )

    ce = a["cross_edge_list"]
    if len(ce) == 0:
        sent.append(f"No ties cross {attr} groups (complete separation).")
    elif len(ce) <= 4:
        brokers = "; ".join(f"{u}-{v}" for u, v in ce)
        plural = "s" if len(ce) != 1 else ""
        verb = "run" if len(ce) != 1 else "runs"
        sent.append(f"The {len(ce)} cross-{attr} tie{plural} {verb} through: {brokers}.")

    mb = a["mean_betweenness_by_group"]
    if len(mb) >= 2:
        hi = max(mb, key=mb.get)
        lo = min(mb, key=mb.get)
        if mb[lo] > 0 and mb[hi] / mb[lo] >= CENTRALITY_RATIO:
            sent.append(f"'{hi}' nodes sit in more central positions "
                        f"({mb[hi] / mb[lo]:.1f}x the mean betweenness of '{lo}').")
        elif mb[hi] > 0 and mb[lo] == 0:
            sent.append(f"'{hi}' nodes occupy the central/brokering positions; "
                        f"'{lo}' nodes are peripheral.")
    return " ".join(sent)


def verbalize_node(G, v):
    """Deterministic description of a SINGLE node (by name/id). No LLM."""
    if v not in G:
        return f"Node {v!r} is not in the graph."
    n = G.number_of_nodes()
    deg = G.degree(v)
    degs = dict(G.degree())
    rank = 1 + sum(1 for u in degs if degs[u] > deg)
    s = [f"Node '{v}': degree {deg} (rank {rank} of {n}), "
         f"local clustering {nx.clustering(G, v):.2f}."]
    if 2 < n <= 1500:
        s.append(f"Betweenness {nx.betweenness_centrality(G).get(v, 0.0):.3f}.")
    try:
        s.append(f"Eigenvector centrality {nx.eigenvector_centrality_numpy(G).get(v, 0.0):.3f}.")
    except Exception:
        pass
    nbrs = [str(u) for u in G.neighbors(v)]
    shown = ", ".join(nbrs[:12]) + ("..." if len(nbrs) > 12 else "")
    s.append(f"Neighbors ({len(nbrs)}): {shown}.")
    return " ".join(s)


def verbalize(facts, focus="all", groups=ALL_GROUPS):
    """Render facts to deterministic text.

    `groups` selects which canonical feature groups (A-K) to render — the single
    knob for ablations; the rendered text then matches feature_vector(facts, groups).
    `focus` restricts the section: 'all' (structure + nodes + attributes),
    'structure' (A-H, J), 'nodes' (I, K), or 'attributes'.
    """
    groups = set(groups)
    parts = []

    if focus in ("all", "structure"):
        struct = [_group_line(g, facts) for g in STRUCTURE_GROUPS if g in groups]
        struct = [x for x in struct if x]
        if struct:
            parts.append("\n".join(struct))

    if focus in ("all", "nodes"):
        for g in NODE_GROUPS:
            if g in groups:
                line = _node_group_line(g, facts)
                if line:
                    parts.append(line)

    if focus in ("all", "attributes"):
        for attr, a in facts.get("attributes", {}).items():
            parts.append(_attribute(facts, attr, a))

    return "\n\n".join(parts)
