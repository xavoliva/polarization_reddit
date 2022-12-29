"""
Pre-processing utils
"""

import os
import string
import re
import time
import operator
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import dask.dataframe as dd
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from constants import EVENTS, EVENTS_DIR, MIN_OCCURENCE_FOR_VOCAB


def txt_to_list(data_path: str) -> list(str):
    with open(data_path, "r", encoding="utf-8") as file:
        data = file.read().splitlines()

        return data


def get_files_from_folder(folder_name: str, compression: str = "bz2") -> list(str):
    # return all files as a list
    files = []
    for file in os.listdir(folder_name):
        if file.endswith(f".{compression}"):
            files.append(f"{folder_name}/{file}")

    return sorted(files)


def load_event(event: str) -> pd.DataFrame:
    return pd.read_csv(
        f"{EVENTS_DIR}/{event}.csv",
        usecols=["author", "body", "created_utc", "affiliation"],
    )


def load_data(
    data_path: str,
    year: int,
    months: list[int] = None,
    tokenize: bool = False,
    comp: str = "bz2",
    columns=list[str],
    dev: bool = False,
):
    files = get_files_from_folder(f"{data_path}/{year}", compression=comp)

    if months:
        files = [f for f in files if int(f[-10:-8]) in months]

    print(f"Loading data of {year}...")

    if dev:
        files = files[:1]

    if comp == "bz2":
        data = dd.read_csv(
            files,
            blocksize=None,  # 500e6 = 500MB
            usecols=columns,
            dtype={
                "subreddit": "string",
                "author": "string",
                "body": "string",
            },
        )

        # keep only day
        data["created_utc"] = dd.to_datetime(data["created_utc"], unit="s").dt.date

    elif comp == "parquet":
        data = dd.read_parquet(files, engine="pyarrow", gather_statistics=True)
    else:
        raise NotImplementedError("Compression not allowed.")

    if tokenize:
        print(f"Tokenizing body... (nr_rows = {len(data)})")

        tic = time.perf_counter()

        data["tokens"] = data["body"].apply(
            lambda x: tokenize_post(x, stopwords.words("english"), stemmer=True)
        )
        toc = time.perf_counter()

        print(f"\tTokenized dataframe in {toc - tic:0.4f} seconds")
    if dev:
        return data.sample(frac=0.01)
    return data


def process_post(text: str) -> str:
    # lower case
    text = text.lower()
    # eliminate urls
    text = re.sub(r"http\S*|\S*\.com\S*|\S*www\S*", " ", text)
    # replace all whitespace with a single space
    text = re.sub(r"\s+", " ", text)
    # strip off spaces on either end
    text = text.strip()

    return text


def tokenize_post(text, keep_stopwords=False, stemmer=True):
    p_text = process_post(text)

    tokens = word_tokenize(p_text)
    # filter punctuation and digits
    tokens = filter(
        lambda token: token not in string.punctuation and not token.isdigit(), tokens
    )

    if not keep_stopwords:
        # filter stopwords
        tokens = [t for t in tokens if t not in stopwords.words('english')]
    # stem words
    if stemmer:
        sno = nltk.stem.LancasterStemmer()  # ("english")
        tokens = [sno.stem(t) for t in tokens]

    return tokens


def get_sentiment_score(post):
    # Create a SentimentIntensityAnalyzer object
    sia = SentimentIntensityAnalyzer()

    post = process_post(post)
    return sia.polarity_scores(post)["compound"]


def calculate_user_party(user_comments: pd.Da):
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
        data = pd.read_csv(f"{EVENTS_DIR}/{event}.csv", usecols=["body"])

        # print(e, len(data))
        # sample a (quasi-)equal number of tweets from each event
        # this has to be done to eliminate words that are too specific to a particular event
        data = data.sample(min(len(data), 10000), random_state=seed_val)
        word_counts = Counter(
            tokenize_post(" ".join(data["body"]), keep_stopwords=True)
        )
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
    data = pd.read_csv(f"{EVENTS_DIR}/{event}.csv", usecols=["body"])

    corpus = tokenize_post(" ".join(data["body"]), stemmer=False, keep_stopwords=False)

    vocab = build_vocab(corpus)
    print("Vocab length:", len(vocab))

    return vocab
