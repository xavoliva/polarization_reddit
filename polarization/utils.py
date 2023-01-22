"""
Polarization utils
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing.utils import split_by_party

RNG = np.random.default_rng(42)


def get_party_q(party_counts, exclude_user=None):
    user_sum = party_counts.sum(axis=0)

    if exclude_user:
        user_sum -= party_counts[exclude_user, :]

    total_sum = user_sum.sum()

    return user_sum / total_sum


def get_rho(dem_q, rep_q):
    return (dem_q / (dem_q + rep_q)).transpose()


def calculate_leaveout_polarization(dem_user_token_cnt, rep_user_token_cnt):
    dem_user_total = dem_user_token_cnt.sum(axis=1)
    rep_user_total = rep_user_token_cnt.sum(axis=1)

    # get row-wise distributions
    dem_user_distr = (sp.diags(1 / dem_user_total.A.ravel())).dot(dem_user_token_cnt)
    rep_user_distr = (sp.diags(1 / rep_user_total.A.ravel())).dot(rep_user_token_cnt)

    nr_dem_users = dem_user_token_cnt.shape[0]
    nr_rep_users = rep_user_token_cnt.shape[0]

    # make sure there are no zero rows
    assert set(dem_user_total.nonzero()[0]) == set(range(nr_dem_users))
    # make sure there are no zero rows
    assert set(rep_user_total.nonzero()[0]) == set(range(nr_rep_users))

    dem_q = get_party_q(dem_user_token_cnt)
    rep_q = get_party_q(rep_user_token_cnt)

    dem_user_polarizations = []
    rep_user_polarizations = []

    dem_polarization_sum = 0
    for i in range(nr_dem_users):
        dem_leaveout_q = get_party_q(dem_user_token_cnt, i)
        dem_token_scores = get_rho(dem_leaveout_q, rep_q)

        dem_user_polarization = dem_user_distr[i, :].dot(dem_token_scores)[0, 0]
        dem_user_polarizations.append(dem_user_polarization)

        dem_polarization_sum += dem_user_polarization

    rep_polarization_sum = 0
    for i in range(nr_rep_users):
        rep_leaveout_q = get_party_q(rep_user_token_cnt, i)
        rep_token_scores = 1.0 - get_rho(dem_q, rep_leaveout_q)

        rep_user_polarization = rep_user_distr[i, :].dot(rep_token_scores)[0, 0]
        rep_user_polarizations.append(rep_user_polarization)

        rep_polarization_sum += rep_user_polarization

    dem_total_polarization = 1 / nr_dem_users * dem_polarization_sum
    rep_total_polarization = 1 / nr_rep_users * rep_polarization_sum

    total_pol = 1 / 2 * (dem_total_polarization + rep_total_polarization)

    return total_pol, dem_user_polarizations, rep_user_polarizations


def get_user_token_counts(comments, vocab: set):
    user_tokens = comments.groupby("author", as_index=False).agg({"tokens": " ".join})

    vec = CountVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, vocabulary=vocab
    )

    user_matrix = vec.fit_transform(user_tokens)

    print(user_matrix.shape)

    return user_matrix


def get_polarization(
    comments: pd.DataFrame,
    vocab: dict[str, int],
    default_score: int = 0.5,
    method: str = "leaveout",
):
    """
    Measure polarization.
    event: name of the event
    comments: dataframe with 'body_cleaned' and 'user'
    default_score: default token partisanship score
    """
    # get partisan comments
    dem_comments, rep_comments = split_by_party(comments)

    dem_user_token_cnt = get_user_token_counts(dem_comments, vocab)
    rep_user_token_cnt = get_user_token_counts(rep_comments, vocab)

    dem_user_cnt = dem_user_token_cnt.shape[0]
    rep_user_cnt = rep_user_token_cnt.shape[0]

    if dem_user_cnt < 10 or rep_user_cnt < 10:
        # return these values when there is not enough data to make predictions on
        return (default_score, default_score, dem_user_cnt + rep_user_cnt), ([], [])

    # make the prior neutral (i.e. make sure there are the same number of dem and rep users)
    if dem_user_cnt > rep_user_cnt:
        dem_user_sample = np.array(RNG.sample(range(dem_user_cnt), rep_user_cnt))
        dem_user_token_cnt = dem_user_token_cnt[dem_user_sample, :]
        dem_user_cnt = dem_user_token_cnt.shape[0]
    elif rep_user_cnt > dem_user_cnt:
        rep_user_sample = np.array(RNG.sample(range(rep_user_cnt), dem_user_cnt))
        rep_user_token_cnt = rep_user_token_cnt[rep_user_sample, :]
        rep_user_cnt = rep_user_token_cnt.shape[0]

    assert dem_user_cnt == rep_user_cnt

    all_counts = sp.vstack([dem_user_token_cnt, rep_user_token_cnt])

    wordcounts = all_counts.nonzero()[1]

    # filter words used by fewer than 2 people
    all_counts = all_counts[
        :,
        np.array(
            [
                (np.count_nonzero(wordcounts == i) > 1)
                for i in range(all_counts.shape[1])
            ]
        ),
    ]

    dem_user_token_cnt = all_counts[:dem_user_cnt, :]
    rep_user_token_cnt = all_counts[dem_user_cnt:, :]

    dem_nonzero = set(dem_user_token_cnt.nonzero()[0])
    rep_nonzero = set(rep_user_token_cnt.nonzero()[0])

    # filter users who did not use words from vocab
    dem_user_token_cnt = dem_user_token_cnt[
        np.array([(i in dem_nonzero) for i in range(dem_user_token_cnt.shape[0])]), :
    ]
    rep_user_token_cnt = rep_user_token_cnt[
        np.array([(i in rep_nonzero) for i in range(rep_user_token_cnt.shape[0])]), :
    ]
    if method == "leaveout":
        (
            total_polarization,
            dem_user_polarizations,
            rep_user_polarizations,
        ) = calculate_leaveout_polarization(dem_user_token_cnt, rep_user_token_cnt)

        all_counts = sp.vstack([dem_user_token_cnt, rep_user_token_cnt])
        index = np.arange(all_counts.shape[0])
        RNG.shuffle(index)
        shuffled_all_counts = all_counts[index, :]

        # Calculate polarization with random assignment of users
        random_polarization, _, _ = calculate_leaveout_polarization(
            shuffled_all_counts[: dem_user_token_cnt.shape[0], :],
            shuffled_all_counts[dem_user_token_cnt.shape[0] :, :],
        )

        user_cnt = dem_user_cnt + rep_user_cnt

        return (total_polarization, random_polarization, user_cnt), (
            dem_user_polarizations,
            rep_user_polarizations,
        )


def split_by_day(comments):
    return [(v, k) for k, v in comments.groupby("date")]


def split_by_week(comments):
    return [(v, k) for k, v in comments.groupby(pd.Grouper(key="date", freq="W"))]


def get_polarization_by_time(event_comments, event_vocab, freq="day"):
    if freq == "day":
        comments_time = split_by_day(event_comments)
    elif freq == "week":
        comments_time = split_by_week(event_comments)

    polarization = []
    for date_comments, date in comments_time:
        (total_polarization, random_polarization, user_cnt), _ = get_polarization(
            date_comments,
            event_vocab,
        )
        polarization.append(
            (total_polarization, random_polarization, user_cnt, pd.to_datetime(date))
        )

    return pd.DataFrame(
        polarization,
        columns=["polarization", "random_polarization", "user_cnt", "date"],
    )
