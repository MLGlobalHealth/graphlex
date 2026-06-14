"""Determinism + correctness checks on the office example. Same graph -> same
facts, every run; the load-bearing numbers match hand computation."""
import networkx as nx
from graphlex import facts, verbalize


def office():
    people = {
        "Alice": ("Eng", "snr"), "Bob": ("Eng", "jr"), "Carol": ("Eng", "jr"),
        "Dan": ("Eng", "jr"), "Eve": ("Eng", "snr"),
        "Frank": ("Sales", "snr"), "Grace": ("Sales", "jr"), "Hank": ("Sales", "jr"),
        "Iris": ("Sales", "jr"), "Jack": ("Sales", "snr"),
    }
    edges = [("Alice", "Bob"), ("Alice", "Carol"), ("Alice", "Dan"), ("Bob", "Carol"),
             ("Carol", "Dan"), ("Dan", "Eve"), ("Bob", "Eve"),
             ("Frank", "Grace"), ("Frank", "Hank"), ("Grace", "Iris"), ("Hank", "Jack"),
             ("Iris", "Jack"), ("Grace", "Jack"), ("Alice", "Frank")]
    G = nx.Graph()
    for p, (d, s) in people.items():
        G.add_node(p, dept=d, seniority=s)
    G.add_edges_from(edges)
    return G


def test_exact_counts():
    f = facts(office(), node_attrs=["dept", "seniority"])
    assert f["structure"]["n_nodes"] == 10
    assert f["structure"]["n_edges"] == 14
    d = f["attributes"]["dept"]
    assert d["cross_edges"] == 1                      # only Alice-Frank crosses depts
    assert d["cross_edge_list"] == [("Alice", "Frank")]
    assert d["assortativity"] > 0.6                   # strongly homophilous on dept


def test_seniors_more_central():
    f = facts(office(), node_attrs=["dept", "seniority"])
    mb = f["attributes"]["seniority"]["mean_betweenness_by_group"]
    assert mb["snr"] > mb["jr"]                       # seniors are the brokers


def test_deterministic():
    a = verbalize(facts(office(), node_attrs=["dept", "seniority"]))
    b = verbalize(facts(office(), node_attrs=["dept", "seniority"]))
    assert a == b                                     # no randomness, ever
