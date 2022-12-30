"""
Pre-processing utils
"""

import os
import time
import operator
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import dask.dataframe as dd
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from load.constants import DATA_DIR
from preprocessing.constants import EVENTS, EVENTS_DIR, MIN_OCCURENCE_FOR_VOCAB

def get_files_from_folder(folder_name: str, file_type: str = "bz2") -> list(str):
    """
    returns all file names from a folder as a list
    """
    file_names = []
    for file_name in os.listdir(folder_name):
        if file_name.endswith(f".{file_type}"):
            file_names.append(f"{folder_name}/{file_name}")

    return sorted(file_names)


def load_event_comments(event: str) -> pd.DataFrame:
    """
    Load dataframe from event
    """
    return pd.read_csv(
        f"{EVENTS_DIR}/{event}.csv",
        usecols=["author", "body_cleaned", "created_utc", "party"],
    )


def load_comments_dask(
    year: int,
    months: list[int] = None,
    tokenize: bool = False,
    file_type: str = "bz2",
    columns=list[str],
    dev: bool = False,
) -> dd.DataFrame:
    file_names = get_files_from_folder(
        f"{DATA_DIR}/comments/comments_{year}", file_type=file_type
    )

    if months:
        file_names = [
            file_name
            for file_name in file_names
            if int(file_name.split(".")[0][-2:]) in months
        ]

    print(f"Loading data of {year}...")

    if dev:
        file_names = file_names[:1]

    if file_type == "bz2":
        data = dd.read_csv(
            file_names,
            blocksize=None,  # 500e6 = 500MB
            usecols=columns,
            dtype={
                "subreddit": "string",
                "author": "string",
                "body_cleaned": "string",
            },
        )

        # keep date only
        data["date"] = dd.to_datetime(data["created_utc"], unit="s").dt.date

    elif file_type == "parquet":
        data = dd.read_parquet(file_names, engine="pyarrow", gather_statistics=True)
    else:
        raise NotImplementedError(f"Compression {file_type} not allowed.")

    if tokenize:
        print(f"Tokenizing body... (nr_rows = {len(data)})")

        tic = time.perf_counter()

        data["tokens"] = data["body_cleaned"].apply(
            lambda x: tokenize_comment(x, stopwords.words("english"), stemmer=True)
        )
        toc = time.perf_counter()

        print(f"\tTokenized dataframe in {toc - tic:0.4f} seconds")
    if dev:
        return data.sample(frac=0.01)
    return data


def split_by_party(data):
    """
    split dataframe by party
    """
    return data[data["party"] == "dem"], data[data["party"] == "rep"]


def tokenize_comment(text, remove_stopwords=False, stemmer=True) -> list(str):
    # body_clean is lowercased, without stopwords and URLs, and without
    # character repetition

    tokens = word_tokenize(text)
    # filter out punctuation
    tokens = [token for token in tokens if token.isalnum()]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords.words("english")]
    # stem words
    if stemmer:
        sno = nltk.stem.LancasterStemmer()  # ("english")
        tokens = [sno.stem(t) for t in tokens]

    return tokens


def get_sentiment_score(comment):
    # Create a SentimentIntensityAnalyzer object
    sia = SentimentIntensityAnalyzer()

    return sia.polarity_scores(comment)["compound"]


def calculate_user_party(user_comments: pd.DataFrameGroupBy) -> pd.Series:
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


def get_all_vocabs(seed_val):
    vocabs = []
    for event in EVENTS:
        data = pd.read_csv(f"{EVENTS_DIR}/{event}.csv", usecols=["body_cleaned"])

        # print(e, len(data))
        # sample a (quasi-)equal number of tweets from each event
        # this has to be done to eliminate words that are too specific to a particular event
        data = data.sample(min(len(data), 10000), random_state=seed_val)
        word_counts = Counter(tokenize_comment(" ".join(data["body_cleaned"])))
        vocab = []
        for word, cnt in word_counts.items():
            if cnt >= 10:  # keep words that occur at least 10 times
                vocab.append(word)
        vocabs.append(set(vocab))
    return vocabs


def word_event_count(vocabs):
    word_event_cnt = {}
    for vocab in vocabs:
        for word in vocab:
            if word in word_event_cnt:
                word_event_cnt[word] += 1
            else:
                word_event_cnt[word] = 1
    # Keep all words that occur in at least three events' tweets. Note that we keep stopwords.
    keep = [
        word
        for word, cnt in sorted(
            word_event_cnt.items(), key=operator.itemgetter(1), reverse=True
        )
        if (cnt > 2 and not word.isdigit())
    ]
    print(len(keep))
    return set(keep)


def build_vocab(corpus):
    uni_and_bigrams = Counter(corpus)

    for i in range(1, len(corpus)):
        word = corpus[i]
        prev_word = corpus[i - 1]
        uni_and_bigrams.update([" ".join([prev_word, word])])

    vocab = [
        word
        for word, cnt in sorted(
            uni_and_bigrams.items(), key=operator.itemgetter(1), reverse=True
        )
        if cnt > MIN_OCCURENCE_FOR_VOCAB
    ]

    return vocab


def build_event_vocab(event):
    data = pd.read_csv(f"{EVENTS_DIR}/{event}.csv", usecols=["body_cleaned"])

    corpus = tokenize_comment(" ".join(data["body_cleaned"]), stemmer=False)

    vocab = build_vocab(corpus)
    print("Vocab length:", len(vocab))

    return vocab
