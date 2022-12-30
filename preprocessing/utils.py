"""
Pre-processing utils
"""

import time
import operator
from collections import Counter

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
import dask.dataframe as dd
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from load.constants import DATA_DIR, COMMENT_DTYPES
from preprocessing.constants import EVENTS, EVENTS_DIR, MIN_OCCURENCE_FOR_VOCAB


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
    start_month: int = 1,
    stop_month: int = 12,
    tokenize: bool = False,
    file_type: str = "bz2",
) -> dd.DataFrame:
    comments_folder = f"{DATA_DIR}/comments/comments_{year}"

    print(f"Loading data of {year}...")

    comments_file_names = [
        f"{comments_folder}/comments_{year}-{month:02}.bz2"
        for month in range(start_month, stop_month + 1)
    ]

    if file_type == "bz2":
        comments = dd.read_json(
            comments_file_names,
            compression="bz2",
            orient="records",
            lines=True,
            blocksize=None,  # 500e6 = 500MB
            dtype=COMMENT_DTYPES,
        )

        comments["date"] = dd.to_datetime(comments["created_utc"], unit="s").dt.date

    elif file_type == "parquet":
        comments = dd.read_parquet(
            comments_file_names,
            engine="pyarrow",
            gather_statistics=True,
        )
    else:
        raise NotImplementedError(f"Compression {file_type} not allowed.")

    if tokenize:
        print(f"Tokenizing body... (nr_rows = {len(comments)})")

        tic = time.perf_counter()

        comments["tokens"] = comments["body_cleaned"].apply(tokenize_comment)
        toc = time.perf_counter()

        print(f"\tTokenized dataframe in {toc - tic:0.4f} seconds")

    return comments


def split_by_party(data):
    """
    split dataframe by party
    """
    return data[data["party"] == "dem"], data[data["party"] == "rep"]


def tokenize_comment(comment: str, stemmer: bool = True) -> list[str]:
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
        sno = LancasterStemmer()  # ("english")
        tokens = [sno.stem(token) for token in tokens]

    return tokens


def get_sentiment_score(comment):
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
