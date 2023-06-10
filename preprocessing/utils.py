"""
Pre-processing utils
"""
import json
from typing import Dict, Tuple

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from preprocessing.constants import (
    EVENTS_DIR,
)

sno = LancasterStemmer()


def split_by_party(comments, backend="pandas") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    split dataframe by party
    """
    if backend == "pandas":
        [dem_comments, rep_comments] = [
            (party, g)
            for party, g in comments.groupby(
                "party",
                sort=True,
            )
        ]
    elif backend == "polars":
        partisan_comments = [
            (party, g)
            for party, g in comments.groupby(
                pl.col("party"),
            )
        ]

        [dem_comments, rep_comments] = sorted(
            partisan_comments,
            key=lambda x: x[0],
        )
    else:
        raise ValueError(f"Backend {backend} not supported")

    return (
        dem_comments[1],
        rep_comments[1],
    )


def tokenize_comment(
    comment: str, stemmer: bool = True, filter_stopwords: bool = True
) -> str:
    """
    Tokenize comment

    Note: body_clean is lowercased, without URLs, and without character repetition
    """

    tokens = word_tokenize(comment)

    # filter out punctuation
    tokens = [token for token in tokens if token.isalnum()]

    if filter_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

    # stem words
    if stemmer:
        tokens = [sno.stem(token) for token in tokens]

    return " ".join(tokens)


def load_event_comments(event_name: str, engine: str = "pandas") -> pd.DataFrame:
    """
    Load dataframe from event
    """
    comments_file = f"{EVENTS_DIR}/{event_name}_comments.parquet"

    if engine == "pandas":
        event_comments = pd.read_parquet(
            comments_file,
            engine="pyarrow",
        )
    elif engine == "polars":
        event_comments = pl.read_parquet(
            comments_file,
        ).to_pandas()

    else:
        raise ValueError(f"Engine {engine} not supported")

    return event_comments


def save_event_comments(event_comments: pd.DataFrame, event_name: str):
    """
    Save event dataframe
    """

    event_comments.to_parquet(
        f"{EVENTS_DIR}/{event_name}_comments.parquet",
        engine="pyarrow",
        compression="snappy",
        index=False,
    )


def save_event_vocab(event_vocab: Dict[str, int], event_name: str):
    with open(f"{EVENTS_DIR}/{event_name}_tokens.json", "w", encoding="utf-8") as file:
        json.dump(event_vocab, file)


def load_event_vocab(event_name: str):
    with open(f"{EVENTS_DIR}/{event_name}_tokens.json", "r", encoding="utf-8") as file:
        return json.load(file)


def get_sentiment_score(comment: str) -> float:
    # Create a SentimentIntensityAnalyzer object
    sia = SentimentIntensityAnalyzer()

    return sia.polarity_scores(comment)["compound"]


def calculate_user_party(partisan_comments: pd.DataFrame) -> pd.Series:
    user_party = {}

    dem_cnt = len(partisan_comments[partisan_comments["party"] == "dem"])
    rep_cnt = len(partisan_comments[partisan_comments["party"] == "rep"])
    score = dem_cnt - rep_cnt

    user_party["dem_cnt"] = dem_cnt
    user_party["rep_cnt"] = rep_cnt
    user_party["score"] = score

    if dem_cnt == 0 and rep_cnt == 0:
        user_party["party"] = ""
        return pd.Series(user_party)

    if dem_cnt > 0 and rep_cnt == 0:
        user_party["party"] = "dem"
    elif rep_cnt > 0 and dem_cnt == 0:
        user_party["party"] = "rep"
    elif score > 0:
        user_party["party"] = "mostly_dem"
    elif score < 0:
        user_party["party"] = "mostly_rep"
    else:
        user_party["party"] = ""

    return pd.Series(user_party)


def build_vocab(corpus, ngram_range: Tuple[int, int], min_df: int) -> Dict[str, int]:
    vec = CountVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
    )
    vec.fit(corpus)
    return vec.vocabulary_


def build_term_matrix(
    corpus,
    ngram_range,
    vocab: Dict[str, int],
) -> csr_matrix:
    vec = CountVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        vocabulary=vocab,
    )

    user_matrix: csr_matrix = vec.transform(corpus)  # type: ignore

    return user_matrix


def build_term_vector(
    corpus,
    ngram_range,
    vocab: Dict[str, int],
) -> np.ndarray:
    term_matrix = build_term_matrix(corpus, ngram_range, vocab)

    doc_term_vec = build_term_vector_from_matrix(term_matrix)

    return doc_term_vec


def build_term_vector_from_matrix(term_matrix: csr_matrix) -> np.ndarray:
    return term_matrix.sum(axis=0).A1  # type: ignore
