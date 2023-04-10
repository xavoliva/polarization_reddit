"""
Polarization utils
"""
# import logging
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import minimize
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from preprocessing.utils import split_by_party
from load.constants import SEED


def get_party_q(
    term_cnt_vec,
    total_token_cnt,
    excluded_user_term_cnt_vec=None,
    excluded_user_token_cnt=None,
) -> np.ndarray:
    if not excluded_user_term_cnt_vec is None:
        leaveout_term_cnt_vec = term_cnt_vec - excluded_user_term_cnt_vec
        leaveout_total_token_cnt = total_token_cnt - excluded_user_token_cnt

        return leaveout_term_cnt_vec / leaveout_total_token_cnt

    return term_cnt_vec / total_token_cnt


def get_rho(dem_q: np.ndarray, rep_q: np.ndarray) -> np.ndarray:
    denom = dem_q + rep_q

    # nr_zero_values_denominator = np.count_nonzero(denom == 0)

    # if nr_zero_values_denominator > 5:
    #     logging.warning(
    #         f"A lot of values in the denominator are zero: {nr_zero_values_denominator}"
    #     )

    return np.divide(dem_q, denom, out=np.zeros_like(dem_q), where=dem_q + rep_q != 0)


def calculate_leaveout_polarization(
    dem_user_term_matrix: sp.csr_matrix,
    rep_user_term_matrix: sp.csr_matrix,
) -> Tuple[float, List[float], List[float]]:
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

    dem_term_cnt_vec = dem_user_term_matrix.sum(axis=0).A1
    rep_term_cnt_vec = rep_user_term_matrix.sum(axis=0).A1

    dem_users_term_cnt_vec = dem_user_term_matrix.sum(axis=1).A1
    rep_users_term_cnt_vec = rep_user_term_matrix.sum(axis=1).A1

    dem_total_token_cnt = dem_term_cnt_vec.sum()
    rep_total_token_cnt = rep_term_cnt_vec.sum()

    dem_q = get_party_q(
        dem_term_cnt_vec,
        dem_total_token_cnt,
    )
    rep_q = get_party_q(
        rep_term_cnt_vec,
        rep_total_token_cnt,
    )

    # Republican polarization
    dem_user_polarizations = []

    for i in tqdm(range(nr_dem_users), desc="Democrat polarization"):
        dem_user_term_vec = dem_user_term_matrix[i].todense().A1  # type: ignore
        dem_leaveout_q = get_party_q(
            dem_term_cnt_vec,
            dem_total_token_cnt,
            dem_user_term_vec,
            dem_users_term_cnt_vec[i],
        )
        dem_token_scores = get_rho(dem_leaveout_q, rep_q)

        dem_user_term_freq_vec = dem_user_term_freq_matrix[i].todense().A1  # type: ignore

        dem_user_polarization = dem_user_term_freq_vec.dot(dem_token_scores)

        dem_user_polarizations.append(dem_user_polarization)

    # Republican polarization
    rep_user_polarizations = []

    for j in tqdm(range(nr_rep_users), desc="Republican polarization"):
        rep_user_term_vec = rep_user_term_matrix[j].todense().A1  # type: ignore
        rep_leaveout_q = get_party_q(
            rep_term_cnt_vec,
            rep_total_token_cnt,
            rep_user_term_vec,
            rep_users_term_cnt_vec[j],
        )
        rep_token_scores = 1.0 - get_rho(dem_q, rep_leaveout_q)

        rep_user_term_freq_vec = rep_user_term_freq_matrix[j].todense().A1

        rep_user_polarization = rep_user_term_freq_vec.dot(rep_token_scores)
        rep_user_polarizations.append(rep_user_polarization)

    dem_total_polarization = np.mean(dem_user_polarizations).item()
    rep_total_polarization = np.mean(rep_user_polarizations).item()

    total_pol = 0.5 * (dem_total_polarization + rep_total_polarization)

    return total_pol, dem_user_polarizations, rep_user_polarizations


def build_user_term_matrix(comments, vocab: Dict[str, int]) -> sp.csr_matrix:
    user_tokens = comments.groupby("author", as_index=False).agg({"tokens": " ".join})

    vec = CountVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, vocabulary=vocab
    )

    user_matrix: sp.csr_matrix = vec.transform(user_tokens["tokens"])  # type: ignore

    return user_matrix


def calculate_penalized_polarization():
    nr_users = 200
    len_vocab = 100
    nr_features = 10

    X = np.ones(shape=(nr_users, nr_features))
    Params0 = np.ones(shape=(len_vocab * (nr_features + 2)), dtype="float32")

    mi = np.ones(shape=(nr_users, 1))

    cij = np.ones(shape=(nr_users, len_vocab))

    is_republican = np.ones(shape=(nr_users, 1))

    lambda_1 = 10e-5
    lambda_2 = 0.3 * np.ones(shape=(len_vocab))

    def penalized_mle(parameters):
        # parameters = (len_vocab * (nr_features + 2))
        # alpha = (len_vocab, 1)
        # gamma = (len_vocab, nr_features)
        # phi = (len_vocab, 1)
        # X = (nr_users, nr_features)
        # mi = (nr_users, 1)
        # cij = (nr_users, len_vocab)
        # is_republican = (nr_users, 1)

        # u = (nr_users, len_vocab)

        alpha = parameters[:len_vocab]
        Gamma = parameters[len_vocab : len_vocab + len_vocab * nr_features].reshape(
            (len_vocab, nr_features)
        )
        phi = parameters[-len_vocab:]

        u = alpha + X @ Gamma.T + np.outer(is_republican, phi)

        out = np.sum(
            mi * np.exp(u)
            - cij * u
            + phi * lambda_1 * (np.abs(alpha) + np.sum(np.abs(Gamma), axis=1))
            + lambda_2 * np.abs(phi)
        )

        return out

    mle_model = minimize(
        fun=penalized_mle,
        x0=Params0,
        method="L-BFGS-B",
        options={
            "maxiter": 100,
            "maxcor": 1000,
            "disp": True,
        },
    )

    print(mle_model)

    return mle_model["x"]


def calculate_polarization(
    comments: pd.DataFrame,
    vocab: Dict[str, int],
    method: str = "leaveout",
) -> Tuple[Tuple[float, float, int], Tuple[List[float], List[float]]]:
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
    dem_user_term_matrix: sp.csr_matrix = dem_user_term_matrix[:, word_ind]  # type: ignore
    rep_user_term_matrix: sp.csr_matrix = rep_user_term_matrix[:, word_ind]  # type: ignore

    # filter out users who did not use words from vocab
    dem_user_term_matrix: sp.csr_matrix = dem_user_term_matrix[
        dem_user_term_matrix.getnnz(axis=1) > 0
    ]  # type: ignore
    rep_user_term_matrix: sp.csr_matrix = rep_user_term_matrix[
        rep_user_term_matrix.getnnz(axis=1) > 0
    ]  # type: ignore

    user_term_matrix: sp.csr_matrix = sp.vstack(
        [
            dem_user_term_matrix,
            rep_user_term_matrix,
        ]
    )  # type: ignore

    dem_user_cnt: int = dem_user_term_matrix.shape[0]
    rep_user_cnt: int = rep_user_term_matrix.shape[0]

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
        dem_user_term_matrix: sp.csr_matrix = dem_user_term_matrix[random_ind]  # type: ignore

    elif rep_user_cnt > dem_user_cnt:
        random_ind = RNG.choice(rep_user_cnt, replace=False, size=dem_user_cnt)
        rep_user_term_matrix: sp.csr_matrix = rep_user_term_matrix[random_ind]  # type: ignore

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

        random_users_dem: sp.csr_matrix = user_term_matrix[
            shuffled_user_ind[: int(user_cnt / 2)]
        ]  # type: ignore
        random_users_rep: sp.csr_matrix = user_term_matrix[
            shuffled_user_ind[int(user_cnt / 2) :]
        ]  # type: ignore

        random_polarization, _, _ = calculate_leaveout_polarization(
            random_users_dem,
            random_users_rep,
        )

        return (total_polarization, random_polarization, user_cnt), (
            dem_user_polarizations,
            rep_user_polarizations,
        )

    else:
        raise ValueError(f"Method {method} not recognized")


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

    polarization_time = pd.DataFrame(
        polarization,
        columns=[
            "polarization",
            "random_polarization",
            "user_cnt",
            "date",
        ],
    )

    polarization_time = polarization_time.astype(
        {
            "polarization": "float",
            "random_polarization": "float",
            "user_cnt": "int",
            "date": "string",
        }
    )

    polarization_time["date"] = pd.to_datetime(
        polarization_time["date"],
    )

    return polarization_time
