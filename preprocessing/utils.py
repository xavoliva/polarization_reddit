"""
Pre-processing utils
"""
import json

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import dask.dataframe as dd
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from preprocessing.constants import (
    EVENTS_DIR,
)


sno = LancasterStemmer()  # ("english")


def split_by_party(data):
    """
    split dataframe by party
    """
    return data[data["party"] == "dem"], data[data["party"] == "rep"]


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


def load_event_comments(
    event_name: str, file_type: str = "parquet"
) -> (pd.DataFrame | dd.DataFrame):
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
        event_comments = dd.read_parquet(
            comments_file,
            engine="pyarrow",
        )
    else:
        raise NotImplementedError(f"File type {file_type} not allowed.")

    return event_comments


def save_event_comments(event_comments, event_name: str, file_type: str = "parquet"):
    """
    Save event dataframe
    """
    if file_type == "csv":
        event_comments.to_csv(
            f"{EVENTS_DIR}/{event_name}_comments.parquet",
            engine="pyarrow",
            compression="snappy",
        )
    elif file_type == "parquet":
        event_comments.to_parquet(
            f"{EVENTS_DIR}/{event_name}_comments.parquet",
            engine="pyarrow",
            compression="snappy",
        )
    else:
        raise NotImplementedError(f"File type {file_type} not allowed.")


def save_event_vocab(event_vocab: dict[str, int], event_name: str):
    with open(f"{EVENTS_DIR}/{event_name}_tokens.json", "w", encoding="utf-8") as fp:
        json.dump(event_vocab, fp)


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


def build_vocab(corpus: pd.Series, min_words: int) -> dict[str, int]:
    vec = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_words)
    vec.fit_transform(corpus)
    return vec.vocabulary_
