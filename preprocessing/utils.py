"""
Pre-processing utils
"""
import json
from typing import Dict, Tuple

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from preprocessing.constants import (
    EVENTS_DIR,
)

pd.options.mode.dtype_backend = 'pyarrow'


sno = LancasterStemmer()


def split_by_party(comments) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    split dataframe by party
    """
    return (
        comments[comments["party"] == "dem"],
        comments[comments["party"] == "rep"],
    )


def tokenize_comment(comment: str, stemmer: bool = True) -> str:
    """
    Tokenize comment

    Note: body_clean is lowercased, without stopwords and URLs, and without
    character repetition
    """

    tokens = word_tokenize(comment)

    # filter out punctuation
    tokens = [token for token in tokens if token.isalnum()]

    # stem words
    if stemmer:
        tokens = [sno.stem(token) for token in tokens]

    return " ".join(tokens)


def load_event_comments(event_name: str, file_type: str = "parquet") -> pd.DataFrame:
    """
    Load dataframe from event
    """
    comments_file = f"{EVENTS_DIR}/{event_name}_comments"

    if file_type == "csv":
        event_comments = pd.read_csv(
            f"{comments_file}.csv",
            usecols=["author", "body_cleaned", "created_utc", "party"],
        )
    elif file_type == "parquet":
        event_comments = pd.read_parquet(
            f"{comments_file}.parquet",
            engine="pyarrow",
        )
    else:
        raise NotImplementedError(f"File type {file_type} not allowed.")

    return event_comments


def save_event_comments(
    event_comments: pd.DataFrame, event_name: str
):
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


def calculate_user_party(user_comments) -> pd.Series:
    user_party = {}

    dem_cnt = len(user_comments[user_comments["party"] == "dem"])
    rep_cnt = len(user_comments[user_comments["party"] == "rep"])
    score = dem_cnt - rep_cnt

    user_party["dem_cnt"] = dem_cnt
    user_party["rep_cnt"] = rep_cnt
    user_party["score"] = score

    if score > 0:
        user_party["party"] = "dem"
    elif score < 0:
        user_party["party"] = "rep"
    else:
        user_party["party"] = ""

    return pd.Series(user_party)


def build_vocab(corpus: pd.Series, min_comment_freq: int) -> Dict[str, int]:
    vec = CountVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_comment_freq,
    )
    vec.fit(corpus)
    return vec.vocabulary_


def build_term_vector(corpus: pd.Series, vocabulary) -> csr_matrix:
    vec = CountVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        vocabulary=vocabulary,
    )
    doc_term_vec = vec.transform([corpus.str.cat(sep=" ")])
    return doc_term_vec
