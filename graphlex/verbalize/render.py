"""verbalize(facts) -> deterministic prose. Templates + threshold tables, NO LLM.

Every sentence is a fixed template filled from exact numbers in `facts`; the only
non-numeric choices come from graphlex.thresholds (versioned, inspectable).
"""
import networkx as nx
from ..thresholds import homophily_word, correspondence_word, CENTRALITY_RATIO


def _fmt(groups):
    return ", ".join(str(g) for g in groups)


def _p(n, noun):
    """'1 node' / '2 nodes' — correct singular/plural."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _assort_phrase(a):
    if a != a:  # NaN
        return "undefined"
    if a >= 0.10:
        return f"{a:+.2f} (assortative: hubs link to hubs)"
    if a <= -0.10:
        return f"{a:+.2f} (disassortative: hubs link to low-degree nodes)"
    return f"{a:+.2f} (neutral)"


def _structure(f):
    s = f["structure"]
    out = (
        f"Network: {_p(s['n_nodes'], 'node')}, {_p(s['n_edges'], 'edge')} "
        f"(density {s['density']:.2f}), {_p(s['n_components'], 'connected component')}. "
        f"Mean degree {s['mean_degree']:.1f} (max {s['max_degree']}, "
        f"std {s['degree_std']:.1f}, max/mean {s['max_over_mean_degree']:.1f}). "
        f"Average clustering {s['avg_clustering']:.2f}, transitivity {s['transitivity']:.2f}. "
        f"Degree assortativity {_assort_phrase(s['degree_assortativity'])}. "
    )
    if s.get("avg_path_length") is not None:
        out += (f"Largest component: average path length {s['avg_path_length']:.2f}, "
                f"diameter {s['diameter']}. ")
    if "n_cycles" in s:
        rs = s.get("ring_sizes")
        if rs:
            ringstr = ", ".join(f"{sz}-rings x{c}" for sz, c in rs.items())
            out += f"{_p(s['n_cycles'], 'independent cycle')} ({ringstr}). "
        elif rs == {}:
            out += f"{_p(s['n_cycles'], 'independent cycle')} (acyclic or no small rings). "
        else:
            out += f"{_p(s['n_cycles'], 'independent cycle')}. "
    out += (f"{s['n_communities']} communit{'y' if s['n_communities']==1 else 'ies'} "
            f"detected.")
    return out


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


def _nodes(f):
    """Named per-node highlights: top nodes by degree / betweenness / eigenvector,
    plus isolates. Empty string when facts() was called with nodes=False."""
    nd = f.get("nodes")
    if not nd or not nd.get("degree"):
        return ""
    deg, bc, eig = nd["degree"], nd["betweenness"], nd["eigenvector"]
    sent = ["Most connected: " + ", ".join(f"{v} (degree {deg[v]})" for v in nd["top_degree"]) + "."]
    if any(x > 0 for x in bc.values()):
        sent.append("Top brokers (betweenness): "
                    + ", ".join(f"{v} ({bc[v]:.2f})" for v in nd["top_betweenness"]) + ".")
    if any(x > 0 for x in eig.values()):
        sent.append("Most central (eigenvector): "
                    + ", ".join(f"{v} ({eig[v]:.2f})" for v in nd["top_eigenvector"]) + ".")
    if nd.get("n_isolates"):
        sent.append(f"{_p(nd['n_isolates'], 'isolated node')} (degree 0).")
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


def verbalize(facts, focus="all"):
    """Render facts to text. focus in {'all','structure','nodes','attributes'}."""
    parts = []
    if focus in ("all", "structure"):
        parts.append(_structure(facts))
    if focus in ("all", "nodes"):
        nt = _nodes(facts)
        if nt:
            parts.append(nt)
    if focus in ("all", "attributes"):
        for attr, a in facts.get("attributes", {}).items():
            parts.append(_attribute(facts, attr, a))
    return "\n\n".join(parts)
