import networkx as nx
import matplotlib.pyplot as plt
import math


def filter_node(
    node,
    network,
    party_subreddits,
    opposition_subreddits,
    weighted,
    threshold=None,
):
    if node in party_subreddits:
        return True

    for opposition_subreddit in opposition_subreddits:
        if network.has_edge(node, opposition_subreddit):
            if not weighted or (
                weighted
                and network.get_edge_data(node, opposition_subreddit)["weight"]
                > threshold
            ):
                return False

    for party_subreddit in party_subreddits:
        if network.has_edge(node, party_subreddit):
            if not weighted or (
                weighted
                and network.get_edge_data(node, party_subreddit)["weight"] > threshold
            ):
                return True

    return False


def draw_network(network, weighted, threshold=None):

    pos = nx.spring_layout(network, k=10/math.sqrt(network.order()), seed=7)

    pos = nx.nx_pydot.graphviz_layout(network)

    # nodes
    nx.draw_networkx_nodes(network, pos, node_size=50)

    # node labels
    nx.draw_networkx_labels(network, pos, font_size=20, font_family="sans-serif")

    if weighted:
        # edge weight labels
        filtered_edges = {
            edge: weight
            for edge, weight in nx.get_edge_attributes(network, "weight").items()
            if weight >= threshold
        }
        # edges
        nx.draw_networkx_edges(network, pos, width=0.5, edgelist=filtered_edges)
        nx.draw_networkx_edge_labels(
            network,
            pos,
            edge_labels=filtered_edges,
        )
    else:
        nx.draw_networkx_edges(network, pos, width=0.5)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
