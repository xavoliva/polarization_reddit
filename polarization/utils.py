"""
Polarization utils
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing.utils import split_by_party

RNG = np.random.default_rng(42)


def get_party_q(user_term_matrix: sp.csr_matrix, excluded_user=None) -> np.ndarray:
    if excluded_user:
        user_term_matrix[excluded_user] = 0

    term_cnt = user_term_matrix.sum(axis=0).A1

    total_token_cnt = user_term_matrix.sum()

    return term_cnt / total_token_cnt


def get_rho(dem_q: np.ndarray, rep_q: np.ndarray) -> np.ndarray:

    assert (
        np.count_nonzero(dem_q + rep_q) < 1
    ), f"{np.count_nonzero(dem_q)}, {np.count_nonzero(rep_q)}"

    return dem_q / (dem_q + rep_q)


def calculate_leaveout_polarization(
    dem_user_term_matrix: sp.csr_matrix,
    rep_user_term_matrix: sp.csr_matrix,
):
    dem_user_term_cnt = dem_user_term_matrix.sum(axis=1).A1
    rep_user_term_cnt = rep_user_term_matrix.sum(axis=1).A1

    # make sure there are no zero rows
    assert np.all(dem_user_term_cnt > 0)
    # make sure there are no zero rows
    assert np.all(rep_user_term_cnt > 0)

    # get row-wise distributions
    dem_user_term_freq_matrix = sp.diags(1 / dem_user_term_cnt) @ dem_user_term_matrix
    rep_user_term_freq_matrix = sp.diags(1 / rep_user_term_cnt) @ rep_user_term_matrix

    nr_dem_users = dem_user_term_matrix.shape[0]
    nr_rep_users = rep_user_term_matrix.shape[0]

    dem_q = get_party_q(dem_user_term_matrix)
    rep_q = get_party_q(rep_user_term_matrix)

    # Republican polarization
    dem_user_polarizations = []
    dem_polarization_sum = 0

    for i in range(nr_dem_users):
        dem_leaveout_q = get_party_q(dem_user_term_matrix, i)
        dem_token_scores = get_rho(dem_leaveout_q, rep_q)

        dem_user_term_freq_vec = dem_user_term_freq_matrix[i].todense().A1

        dem_user_polarization = dem_user_term_freq_vec.dot(dem_token_scores)

        dem_user_polarizations.append(dem_user_polarization)

        dem_polarization_sum += dem_user_polarization

    # Republican polarization
    rep_user_polarizations = []
    rep_polarization_sum = 0

    for i in range(nr_rep_users):
        rep_leaveout_q = get_party_q(rep_user_term_matrix, i)
        rep_token_scores = 1.0 - get_rho(dem_q, rep_leaveout_q)

        rep_user_term_freq_vec = rep_user_term_freq_matrix[i].todense().A1

        rep_user_polarization = rep_user_term_freq_vec.dot(rep_token_scores)
        rep_user_polarizations.append(rep_user_polarization)

        rep_polarization_sum += rep_user_polarization

    dem_total_polarization = 1 / nr_dem_users * dem_polarization_sum
    rep_total_polarization = 1 / nr_rep_users * rep_polarization_sum

    total_pol = 0.5 * (dem_total_polarization + rep_total_polarization)

    return total_pol, dem_user_polarizations, rep_user_polarizations


def build_user_term_matrix(comments, vocab: dict[str, int]):
    user_tokens = comments.groupby("author", as_index=False).agg({"tokens": " ".join})

    vec = CountVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, vocabulary=vocab
    )

    user_matrix = vec.transform(user_tokens["tokens"])

    return user_matrix


def calculate_polarization(
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

    dem_user_term_matrix = build_user_term_matrix(dem_comments, vocab)
    rep_user_term_matrix = build_user_term_matrix(rep_comments, vocab)

    # filter out users who did not use words from vocab
    dem_user_term_matrix = dem_user_term_matrix[dem_user_term_matrix.getnnz(axis=1) > 0]
    rep_user_term_matrix = rep_user_term_matrix[rep_user_term_matrix.getnnz(axis=1) > 0]

    dem_user_cnt = dem_user_term_matrix.shape[0]
    rep_user_cnt = rep_user_term_matrix.shape[0]

    if dem_user_cnt < 10 or rep_user_cnt < 10:
        # return these values when there is not enough data to make predictions on
        raise RuntimeError(
            f"""Not enough data to make predictions:
            - Number of democrat users: {dem_user_cnt}
            - Number of republican users: {rep_user_cnt}
            """
        )

    # make the prior neutral (i.e. make sure there are the same number of dem and rep users)
    if dem_user_cnt > rep_user_cnt:
        random_ind = RNG.choice(dem_user_cnt, size=rep_user_cnt)
        dem_user_term_matrix = dem_user_term_matrix[random_ind]

    elif rep_user_cnt > dem_user_cnt:
        random_ind = RNG.choice(rep_user_cnt, size=dem_user_cnt)
        rep_user_term_matrix = rep_user_term_matrix[random_ind]

    assert dem_user_term_matrix.shape[0] == rep_user_term_matrix.shape[0]

    user_cnt = dem_user_term_matrix.shape[0] + rep_user_term_matrix.shape[0]

    user_term_matrix = sp.vstack(
        (
            dem_user_term_matrix,
            rep_user_term_matrix,
        )
    )

    # filter out words used by fewer than 2 people
    # dem_user_term_matrix = dem_user_term_matrix[:, user_term_matrix.getnnz(axis=0) > 1]
    # rep_user_term_matrix = rep_user_term_matrix[:, user_term_matrix.getnnz(axis=0) > 1]

    if method == "leaveout":
        (
            total_polarization,
            dem_user_polarizations,
            rep_user_polarizations,
        ) = calculate_leaveout_polarization(
            dem_user_term_matrix,
            rep_user_term_matrix,
        )

        # Calculate polarization with random assignment of users

        shuffled_ind = np.arange(user_cnt)
        shuffled_user_token_cnt = user_term_matrix[shuffled_ind]

        random_polarization, _, _ = calculate_leaveout_polarization(
            shuffled_user_token_cnt[: int(user_cnt / 2)],
            shuffled_user_token_cnt[int(user_cnt / 2) :],
        )

        return (total_polarization, random_polarization, user_cnt), (
            dem_user_polarizations,
            rep_user_polarizations,
        )


def split_by_day(comments):
    return [(v, k) for k, v in comments.groupby("date")]


def split_by_week(comments):
    return [(v, k) for k, v in comments.groupby(pd.Grouper(key="date", freq="W"))]


def calculate_polarization_by_time(event_comments, event_vocab, freq="day"):
    if freq == "day":
        comments_time = split_by_day(event_comments)
    elif freq == "week":
        comments_time = split_by_week(event_comments)

    polarization = []
    for date_comments, date in comments_time:
        (total_polarization, random_polarization, user_cnt), _ = calculate_polarization(
            date_comments,
            event_vocab,
        )
        polarization.append(
            (
                total_polarization,
                random_polarization,
                user_cnt,
                pd.to_datetime(date),
            )
        )

    return pd.DataFrame(
        polarization,
        columns=[
            "polarization",
            "random_polarization",
            "user_cnt",
            "date",
        ],
    )
