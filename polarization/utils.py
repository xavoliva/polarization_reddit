"""
Polarization utils
"""
# import warnings
from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from preprocessing.utils import split_by_party
from load.constants import SEED


def get_party_q(user_term_matrix: sp.csr_matrix, excluded_user=None) -> np.ndarray:

    term_cnt = user_term_matrix.sum(axis=0).A1
    total_token_cnt = user_term_matrix.sum()

    if excluded_user:
        user_term_vec = user_term_matrix[excluded_user].todense().A1

        term_cnt -= user_term_vec
        total_token_cnt -= user_term_vec.sum()

    return term_cnt / total_token_cnt


def get_rho(dem_q: np.ndarray, rep_q: np.ndarray) -> np.ndarray:

    denom = dem_q + rep_q

    nr_zero_values_denominator = np.count_nonzero(denom == 0)

    # if nr_zero_values_denominator > 5:
    #     warnings.warn(
    #         f"A lot of values in the denominator are zero: {nr_zero_values_denominator}"
    #     )

    return np.divide(dem_q, denom, out=np.zeros_like(dem_q), where=dem_q + rep_q != 0)


def calculate_leaveout_polarization(
    dem_user_term_matrix: sp.csr_matrix,
    rep_user_term_matrix: sp.csr_matrix,
):
    dem_user_term_cnt = dem_user_term_matrix.sum(axis=1).A1
    rep_user_term_cnt = rep_user_term_matrix.sum(axis=1).A1

    # make sure there are no zero rows
    assert (
        np.count_nonzero(dem_user_term_cnt == 0) == 0
    ), f"{np.count_nonzero(dem_user_term_cnt == 0)} dem users without words!"
    # make sure there are no zero rows
    assert (
        np.count_nonzero(rep_user_term_cnt == 0) == 0
    ), f"{np.count_nonzero(rep_user_term_cnt == 0)} rep users without words!"

    # get row-wise distributions
    dem_user_term_freq_matrix = sp.diags(1 / dem_user_term_cnt) @ dem_user_term_matrix
    rep_user_term_freq_matrix = sp.diags(1 / rep_user_term_cnt) @ rep_user_term_matrix

    nr_dem_users = dem_user_term_matrix.shape[0]
    nr_rep_users = rep_user_term_matrix.shape[0]

    dem_q = get_party_q(dem_user_term_matrix)
    rep_q = get_party_q(rep_user_term_matrix)

    # Republican polarization
    dem_user_polarizations = []

    for i in tqdm(range(nr_dem_users), desc="Democrat polarization"):
        dem_leaveout_q = get_party_q(dem_user_term_matrix, i)
        dem_token_scores = get_rho(dem_leaveout_q, rep_q)

        dem_user_term_freq_vec = dem_user_term_freq_matrix[i].todense().A1

        dem_user_polarization = dem_user_term_freq_vec.dot(dem_token_scores)

        dem_user_polarizations.append(dem_user_polarization)

    # Republican polarization
    rep_user_polarizations = []

    for i in tqdm(range(nr_rep_users), desc="Republican polarization"):
        rep_leaveout_q = get_party_q(rep_user_term_matrix, i)
        rep_token_scores = 1.0 - get_rho(dem_q, rep_leaveout_q)

        rep_user_term_freq_vec = rep_user_term_freq_matrix[i].todense().A1

        rep_user_polarization = rep_user_term_freq_vec.dot(rep_token_scores)
        rep_user_polarizations.append(rep_user_polarization)

    dem_total_polarization = np.mean(dem_user_polarizations)
    rep_total_polarization = np.mean(rep_user_polarizations)

    total_pol = 0.5 * (dem_total_polarization + rep_total_polarization)

    return total_pol, dem_user_polarizations, rep_user_polarizations


def build_user_term_matrix(comments, vocab: Dict[str, int]):
    user_tokens = comments.groupby("author", as_index=False).agg({"tokens": " ".join})

    vec = CountVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, vocabulary=vocab
    )

    user_matrix = vec.transform(user_tokens["tokens"])

    return user_matrix


def calculate_polarization(
    comments: pd.DataFrame,
    vocab: Dict[str, int],
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

    # filter out words used by fewer than 2 people
    word_ind = (
        sp.vstack([dem_user_term_matrix, rep_user_term_matrix]).getnnz(axis=0) > 1
    )
    dem_user_term_matrix = dem_user_term_matrix[:, word_ind]
    rep_user_term_matrix = rep_user_term_matrix[:, word_ind]

    # filter out users who did not use words from vocab
    dem_user_term_matrix = dem_user_term_matrix[dem_user_term_matrix.getnnz(axis=1) > 0]
    rep_user_term_matrix = rep_user_term_matrix[rep_user_term_matrix.getnnz(axis=1) > 0]

    user_term_matrix = sp.vstack([dem_user_term_matrix, rep_user_term_matrix])

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

    RNG = np.random.default_rng(SEED)

    # make the prior neutral (i.e. make sure there are the same number of dem and rep users)
    if dem_user_cnt > rep_user_cnt:
        random_ind = RNG.choice(dem_user_cnt, replace=False, size=rep_user_cnt)
        dem_user_term_matrix = dem_user_term_matrix[random_ind]

    elif rep_user_cnt > dem_user_cnt:
        random_ind = RNG.choice(rep_user_cnt, replace=False, size=dem_user_cnt)
        rep_user_term_matrix = rep_user_term_matrix[random_ind]

    assert dem_user_term_matrix.shape[0] == rep_user_term_matrix.shape[0]

    user_cnt = dem_user_term_matrix.shape[0] + rep_user_term_matrix.shape[0]

    if method == "leaveout":
        print("Calculate leave-out polarization of users")
        (
            total_polarization,
            dem_user_polarizations,
            rep_user_polarizations,
        ) = calculate_leaveout_polarization(
            dem_user_term_matrix,
            rep_user_term_matrix,
        )

        print("Calculate leave-out polarization with random assignment of users")

        shuffled_user_ind = RNG.choice(user_cnt, replace=False, size=user_cnt)

        random_polarization, _, _ = calculate_leaveout_polarization(
            user_term_matrix[shuffled_user_ind[: int(user_cnt / 2)]],
            user_term_matrix[shuffled_user_ind[int(user_cnt / 2) :]],
        )

        return (total_polarization, random_polarization, user_cnt), (
            dem_user_polarizations,
            rep_user_polarizations,
        )


def calculate_polarization_by_time(event_comments, event_vocab, freq="D"):

    polarization = []
    for datetime, date_comments in event_comments.groupby(
        pd.Grouper(key="datetime", freq=freq)
    ):
        print(datetime)
        (total_polarization, random_polarization, user_cnt), _ = calculate_polarization(
            date_comments,
            event_vocab,
        )
        print(total_polarization)
        polarization.append(
            (
                total_polarization,
                random_polarization,
                user_cnt,
                datetime,
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
