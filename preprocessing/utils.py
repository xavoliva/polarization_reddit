"""
Pre-processing utils
"""

import networkx as nx
import pandas as pd


def load_network(year: int, weighted=False) -> nx.Graph:
    """Load subreddits network of given year

    Args:
        year (int): Year
        weighted (str): weighted vs unweighted

    Returns:
        nx.Graph: networkx graph
    """
    network_file = f"/workspaces/polarization_reddit/data/networks/networks_{year}.csv"

    # Load network
    network = pd.read_csv(network_file)

    # Extract weighted or unweighted network
    if weighted:
        graph = nx.Graph()
        edges = zip(network["node_1"], network["node_2"], network["weighted"])
        graph.add_weighted_edges_from(edges)
    else:
        graph = nx.Graph()
        edges = zip(
            network[network.unweighted == 1]["node_1"],
            network[network.unweighted == 1]["node_2"],
        )
        graph.add_edges_from(edges)

    return graph


def load_comments(
    year: int, start_month: int = 1, stop_month: int = 12
) -> pd.DataFrame:
    """Load all comments in a given year

    Args:
        year (int): year
        months (list[int], optional): Range of months to load. Defaults to None.

    Returns:
        pd.DataFrame: comments pandas dataframe
    """
    comments_folder = f"/workspaces/polarization_reddit/data/comments/comments_{year}"
    comments = []
    # Load comments in chunks
    for month in range(start_month, stop_month + 1):
        comments_file = f"{comments_folder}/comments_{year}-{month:02}.bz2"
        for comments_chunk_df in pd.read_json(
            comments_file,
            compression="bz2",
            lines=True,
            dtype=False,
            chunksize=1e4,
        ):
            comments_chunk_df = comments_chunk_df[
                (comments_chunk_df.body != "[deleted]")
            ]
            comments.append(comments_chunk_df)

    df_comments = pd.concat(comments, ignore_index=True)

    return df_comments


def load_users() -> pd.DataFrame:
    """Load all users

    Returns:
        pd.DataFrame: user dataframe
    """
    users = []
    for users_chunk_df in pd.read_json(
        "/workspaces/polarization_reddit/data/metadata/users_metadata.json",
        orient="records",
        lines=True,
        chunksize=1e4,
    ):
        users_chunk_df = users_chunk_df[
            (users_chunk_df.bot == 0) & (users_chunk_df.automoderator == 0)
        ]
        users.append(users_chunk_df)

    df_users = pd.concat(users, ignore_index=True)

    return df_users


def save_df_as_json(data: pd.DataFrame, target_file):
    folder = "/workspaces/polarization_reddit/data/output"
    data.to_json(f"{folder}/{target_file}", orient="records", lines=True)
