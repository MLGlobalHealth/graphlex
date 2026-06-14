"""verbalize(facts) -> deterministic prose. Templates + threshold tables, NO LLM.

Every sentence is a fixed template filled from exact numbers in `facts`; the only
non-numeric choices come from graphlex.thresholds (versioned, inspectable).
"""
from ..thresholds import homophily_word, correspondence_word, CENTRALITY_RATIO


def _fmt(groups):
    return ", ".join(str(g) for g in groups)


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
        f"Network: {s['n_nodes']} nodes, {s['n_edges']} edges "
        f"(density {s['density']:.2f}), {s['n_components']} connected component(s). "
        f"Mean degree {s['mean_degree']:.1f} (max {s['max_degree']}, "
        f"std {s['degree_std']:.1f}, max/mean {s['max_over_mean_degree']:.1f}). "
        f"Average clustering {s['avg_clustering']:.2f}, transitivity {s['transitivity']:.2f}. "
        f"Degree assortativity {_assort_phrase(s['degree_assortativity'])}. "
    )
    if s.get("avg_path_length") is not None:
        out += (f"Largest component: average path length {s['avg_path_length']:.2f}, "
                f"diameter {s['diameter']}. ")
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


def verbalize(facts, focus="all"):
    """Render facts to text. focus in {'all','structure','attributes'}."""
    parts = []
    if focus in ("all", "structure"):
        parts.append(_structure(facts))
    if focus in ("all", "attributes"):
        for attr, a in facts.get("attributes", {}).items():
            parts.append(_attribute(facts, attr, a))
    return "\n\n".join(parts)
