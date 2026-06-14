"""Inspectable, versioned threshold tables that map exact numbers to words.

This is the ONLY place "interpretation" happens. It is data, not a model: edit a
number here and the wording changes everywhere, reproducibly. No LLM.
"""

# attribute assortativity (homophily) -> phrase. Checked high-to-low.
HOMOPHILY = [
    (0.60, "strongly homophilous"),
    (0.30, "moderately homophilous"),
    (0.10, "weakly homophilous"),
    (-0.10, "mixed (no homophily)"),
    (-1.01, "heterophilous (ties tend to cross groups)"),
]

# community<->attribute purity (fraction of nodes matching their community's
# majority attribute value) -> phrase. Checked high-to-low.
CORRESPONDENCE = [
    (0.90, "almost exactly"),
    (0.75, "closely"),
    (0.60, "loosely"),
    (0.00, "only weakly"),
]

# ratio of group mean betweenness considered "notably more central".
CENTRALITY_RATIO = 1.5


def homophily_word(a: float) -> str:
    if a != a:  # NaN (e.g. single group)
        return "undefined (only one group present)"
    for thr, word in HOMOPHILY:
        if a >= thr:
            return word
    return "heterophilous (ties tend to cross groups)"


def correspondence_word(p: float) -> str:
    for thr, word in CORRESPONDENCE:
        if p >= thr:
            return word
    return "not at all"
