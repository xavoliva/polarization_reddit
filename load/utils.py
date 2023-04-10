"""
Pre-processing utils
"""

from typing import List, Union
import os

import networkx as nx
import pandas as pd
import polars as pl
from tqdm import tqdm

from load.constants import (
    COMMENT_COLUMNS,
    COMMENT_DTYPES,
    DATA_DIR,
    SUBREDDIT_DTYPES,
    USER_DTYPES,
    USER_COLUMNS,
)

from preprocessing.constants import OUTPUT_DIR


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
    years: Union[int, List[int]],
    start_month: int = 1,
    stop_month: int = 12,
    engine="pandas",
) -> pd.DataFrame:
    """Load all comments in a given year
    Args:
        year (int): year
        start_month (int): start month
        stop_month (int): stop month (included)

    Returns:
        DataFrame: comments dataframe
    """
    if isinstance(years, int):
        years = [years]

    # Load comments in chunks
    if engine == "pandas":
        comments = []
        for i, year in enumerate(years):
            print(year)
            comments_folder = f"{DATA_DIR}/comments/comments_{year}"

            months = range(
                start_month if i == 0 else 1,
                stop_month + 1 if i == len(years) - 1 else 12 + 1,
            )
            for month in tqdm(months, desc="Months"):
                comments_file_name = (
                    f"{comments_folder}/comments_{year}-{month:02}.json"
                )
                compression = None
                # if file does not exist
                if not os.path.exists(comments_file_name):
                    comments_file_name = comments_file_name.replace("json", "bz2")
                    compression = "bz2"

                comments_month = pd.read_json(
                    comments_file_name,
                    compression=compression,
                    orient="records",
                    lines=True,
                    dtype=COMMENT_DTYPES,  # type: ignore
                    # chunksize=1e4,
                )

                comments_month = comments_month[
                    (comments_month.body != "[deleted]")
                    & (comments_month.author != "[deleted]")
                    & (comments_month.language == "en")
                ][COMMENT_COLUMNS]

                comments_month["created_utc"] = comments_month["created_utc"].astype(
                    "int64"
                )

                comments.append(comments_month)

        df_comments = pd.concat(comments, ignore_index=True)

        return df_comments

    elif engine == "polars":
        queries = []
        for i, year in enumerate(years):
            comments_folder = f"{DATA_DIR}/comments/comments_{year}"
            months = range(
                start_month if i == 0 else 1,
                stop_month + 1 if i == len(years) - 1 else 12 + 1,
            )

            for month in months:
                comments_file_name = (
                    f"{comments_folder}/comments_{year}-{month:02}.json"
                )

                q = pl.scan_ndjson(comments_file_name)

                queries.append(q)

        df_list = pl.collect_all(queries)

        df_pl = pl.concat(
            [
                df.filter(
                    (pl.col("body") != "[deleted]")
                    & (pl.col("author") != "[deleted]")
                    & (pl.col("language") == "en")
                ).select(COMMENT_COLUMNS)
                for df in df_list
            ]
        )

        return df_pl.to_pandas().astype(
            {
                "author": "string[pyarrow]",
                "body_cleaned": "string[pyarrow]",
                "created_utc": "int64[pyarrow]",
                "subreddit": "string[pyarrow]",
            }
        )

    else:
        raise ValueError("Engine not supported")


def load_users(engine) -> pd.DataFrame:
    """Load all users

    Returns:
        pd.DataFrame: user dataframe
    """
    users_file_name = f"{DATA_DIR}/metadata/users_metadata.json"
    if engine == "pandas":
        df = pd.read_json(
            users_file_name,
            orient="records",
            lines=True,
            dtype_backend="pyarrow",
            dtype=USER_DTYPES,  # type: ignore
        )
        df = df[~(df.bot) & ~(df.automoderator)]
        return df[USER_COLUMNS]

    elif engine == "polars":
        df_pl = pl.read_ndjson(users_file_name)

        df_pl = df_pl.filter(~(pl.col("bot") == 1) & ~(pl.col("automoderator") == 1))

        return df_pl.to_pandas().astype(USER_DTYPES)[USER_COLUMNS]

    else:
        raise ValueError("Engine not supported")


def load_subreddits() -> pd.DataFrame:
    """Load all subreddits

    Returns:
        pd.DataFrame: subreddit dataframe
    """
    subreddits = pd.read_json(
        f"{DATA_DIR}/metadata/subreddits_metadata.json",
        orient="records",
        lines=True,
        dtype=SUBREDDIT_DTYPES,  # type: ignore
    )

    # Filter out regional and international subreddits
    subreddits = subreddits[(subreddits["region"] == "")]

    return subreddits


def load_txt_to_list(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read().splitlines()

        return data


def save_df_as_json(data: pd.DataFrame, target_file: str):
    data.to_json(
        f"{OUTPUT_DIR}/{target_file}",
        orient="records",
        lines=True,
        index=False,
    )


def save_df_as_parquet(data: pd.DataFrame, target_file: str):
    data.to_parquet(
        f"{OUTPUT_DIR}/{target_file}",
        engine="pyarrow",
        compression="snappy",
        index=False,
    )


def load_df_from_parquet(file_name: str) -> pd.DataFrame:
    return pd.read_parquet(
        f"{OUTPUT_DIR}/{file_name}",
        engine="pyarrow",
    )
