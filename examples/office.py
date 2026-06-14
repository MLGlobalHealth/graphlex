"""The structure x attributes worked example: a 10-person office collaboration
network, siloed by department, bridged only by senior staff."""
import networkx as nx
from graphlex import facts, verbalize

people = {
    "Alice": ("Eng", "snr"), "Bob": ("Eng", "jr"), "Carol": ("Eng", "jr"),
    "Dan": ("Eng", "jr"), "Eve": ("Eng", "snr"),
    "Frank": ("Sales", "snr"), "Grace": ("Sales", "jr"), "Hank": ("Sales", "jr"),
    "Iris": ("Sales", "jr"), "Jack": ("Sales", "snr"),
}
edges = [
    ("Alice", "Bob"), ("Alice", "Carol"), ("Alice", "Dan"), ("Bob", "Carol"),
    ("Carol", "Dan"), ("Dan", "Eve"), ("Bob", "Eve"),
    ("Frank", "Grace"), ("Frank", "Hank"), ("Grace", "Iris"), ("Hank", "Jack"),
    ("Iris", "Jack"), ("Grace", "Jack"),
    ("Alice", "Frank"),  # the single cross-department tie
]

G = nx.Graph()
for p, (dept, sen) in people.items():
    G.add_node(p, dept=dept, seniority=sen)
G.add_edges_from(edges)

f = facts(G, node_attrs=["dept", "seniority"])
print(verbalize(f, focus="all"))
