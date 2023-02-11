"""
Pre-processing utils
"""

from typing import List, Union

import networkx as nx
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm

from load.constants import DATA_DIR, COMMENT_DTYPES, COMMENT_COLUMNS


def load_network(year: int, weighted=False) -> nx.Graph:
    """Load subreddits network of given year

    Args:
        year (int): year
        weighted (str): weighted vs unweighted

    Returns:
        nx.Graph: networkx graph
    """
    network_file = f"{DATA_DIR}/networks/networks_{year}.csv"

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
        start_month (int): start month
        stop_month (int): stop month (included)

    Returns:
        pd.DataFrame: comments pandas dataframe
    """
    comments_folder = f"{DATA_DIR}/comments/comments_{year}"
    comments = []
    # Load comments in chunks
    for month in tqdm(range(start_month, stop_month + 1), desc="Months"):
        comments_file_name = f"{comments_folder}/comments_{year}-{month:02}.bz2"
        for comments_chunk_df in pd.read_json(
            comments_file_name,
            compression="bz2",
            orient="records",
            lines=True,
            dtype=COMMENT_DTYPES,
            chunksize=1e4,
        ):
            comments_chunk_df = comments_chunk_df[
                (comments_chunk_df.body != "[deleted]")
                & (comments_chunk_df.author != "[deleted]")
                & (comments_chunk_df.language == "en")
            ][COMMENT_COLUMNS]
            comments.append(comments_chunk_df)

    df_comments = pd.concat(comments, ignore_index=True)

    return df_comments


def load_comments_dask(
    year: int,
    start_month: int = 1,
    stop_month: int = 12,
) -> dd.DataFrame:
    comments_folder = f"{DATA_DIR}/comments/comments_{year}"

    comments_file_names = [
        f"{comments_folder}/comments_{year}-{month:02}.bz2"
        for month in range(start_month, stop_month + 1)
    ]

    comments = dd.read_json(
        comments_file_names,
        compression="bz2",
        orient="records",
        lines=True,
        # blocksize=None,  # 500e6 = 500MB
        dtype=COMMENT_DTYPES,
    )

    comments = comments[
        (comments["author"] != "[deleted]")
        & (comments["body"] != "[deleted]")
        & (comments["language"] == "en")
    ][COMMENT_COLUMNS]

    return comments


def load_users() -> pd.DataFrame:
    """Load all users

    Returns:
        pd.DataFrame: user dataframe
    """
    users = []
    for users_chunk_df in pd.read_json(
        f"{DATA_DIR}/metadata/users_metadata.json",
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


def load_user_party(year: int) -> pd.DataFrame:
    """Load user party affiliation

    Returns:
        pd.DataFrame: user-party dataframe
    """
    user_party = pd.read_json(f"{DATA_DIR}/output/user_party_{year}.json", lines=True)

    return user_party


def load_user_party_parquet(year: int) -> dd.DataFrame:
    """Load user party affiliation

    Returns:
        dd.DataFrame: user-party dataframe
    """
    user_party = pd.read_parquet(
        f"{DATA_DIR}/output/user_party_{year}.parquet",
        columns=["author", "party"],
    )

    return user_party


def load_subreddits() -> pd.DataFrame:
    """Load all users

    Returns:
        pd.DataFrame: user dataframe
    """
    subreddits = pd.read_json(
        f"{DATA_DIR}/metadata/subreddits_metadata.json",
        orient="records",
        lines=True,
        dtype={
            "subreddit": "string",
            "banned": "bool",
            "gun": "bool",
            "meta": "bool",
            "party": "string",
            "politician": "bool",
            "region": "string",
        },
    )

    # Filter out regional and international subreddits
    subreddits = subreddits[(subreddits["region"] == "")]

    return subreddits


def load_txt_to_list(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read().splitlines()

        return data


def save_df_as_json(data: Union[pd.DataFrame, dd.DataFrame], target_file: str):
    output_folder = f"{DATA_DIR}/output"
    data.to_json(
        f"{output_folder}/{target_file}",
        orient="records",
        lines=True,
        index=False,
    )


def save_df_as_parquet(data: pd.DataFrame, target_file: str):
    output_folder = f"{DATA_DIR}/output"
    data.to_parquet(
        f"{output_folder}/{target_file}",
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
