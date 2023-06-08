"""
Polarization utils
"""
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from preprocessing.utils import split_by_party, build_vocab
from load.constants import SEED


def get_party_q(
    term_cnt_vec: np.ndarray,
    total_token_cnt: int,
    excluded_user_term_cnt_vec: Optional[np.ndarray] = None,
    excluded_user_token_cnt: Optional[int] = None,
) -> np.ndarray:
    if excluded_user_term_cnt_vec is not None and excluded_user_token_cnt is not None:
        leaveout_term_cnt_vec = term_cnt_vec - excluded_user_term_cnt_vec
        leaveout_total_token_cnt = total_token_cnt - excluded_user_token_cnt

        return np.divide(leaveout_term_cnt_vec, leaveout_total_token_cnt)

    return np.divide(term_cnt_vec, total_token_cnt)


def get_rho(q_1: np.ndarray, q_2: np.ndarray) -> np.ndarray:
    denom = q_1 + q_2

    nr_zero_values_denominator = np.count_nonzero(denom == 0)

    if nr_zero_values_denominator > 0:
        RuntimeError(f"{nr_zero_values_denominator} values in the denominator are zero")

    return np.divide(q_1, denom)


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

    # Democrat polarization
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
        rep_token_scores = get_rho(rep_leaveout_q, dem_q)

        rep_user_term_freq_vec = rep_user_term_freq_matrix[j].todense().A1

        rep_user_polarization = rep_user_term_freq_vec.dot(rep_token_scores)
        rep_user_polarizations.append(rep_user_polarization)

    dem_total_polarization = np.mean(dem_user_polarizations).item()
    rep_total_polarization = np.mean(rep_user_polarizations).item()

    total_pol = 0.5 * (dem_total_polarization + rep_total_polarization)

    return total_pol, dem_user_polarizations, rep_user_polarizations


def build_user_term_matrix(
    user_tokens,
    ngram_range,
    vocab: Dict[str, int],
) -> sp.csr_matrix:
    vec = CountVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        vocabulary=vocab,
    )

    user_matrix: sp.csr_matrix = vec.transform(user_tokens["tokens"])  # type: ignore

    return user_matrix


def calculate_polarization(
    comments: pd.DataFrame,
    ngram_range: Tuple[int, int],
    event_vocab: Dict[str, int],
    method: str = "leaveout",
    equalize_users: bool = True,
) -> Tuple[Tuple[float, float, int], Tuple[List[float], List[float]]]:
    """
    Measure polarization.
    event: name of the event
    comments: dataframe with 'body_cleaned' and 'user'
    default_score: default token partisanship score
    """
    user_tokens = comments.groupby("author", as_index=False).agg(
        tokens=("tokens", lambda x: " ".join(x)),
        party=("party", lambda x: x.iloc[0]),
        nr_comments=("id", "count"),
    )

    if equalize_users:
        print("Equalize users")
        dem_user_tokens, rep_user_tokens = split_by_party(user_tokens)
        (
            dem_user_tokens,
            rep_user_tokens,
        ) = get_balanced_partisan_user_tokens(dem_user_tokens, rep_user_tokens)

        user_tokens = pd.concat([dem_user_tokens, rep_user_tokens])

    user_term_matrix = build_user_term_matrix(user_tokens, ngram_range, event_vocab)

    user_cnt = user_term_matrix.shape[0]

    # filter out words used by fewer than 2 people
    word_ind = [
        ind
        for ind, is_more_than_one in enumerate(
            user_term_matrix.getnnz(axis=0) >= 2  # type: ignore
        )
        if is_more_than_one
    ]

    user_term_matrix: sp.csr_matrix = user_term_matrix[:, word_ind]  # type: ignore

    dem_indices = [
        ind
        for ind, is_dem in enumerate((user_tokens["party"] == "dem").values)
        if is_dem
    ]

    rep_indices = [
        ind
        for ind, is_rep in enumerate((user_tokens["party"] == "rep").values)
        if is_rep
    ]

    dem_user_term_matrix: sp.csr_matrix = user_term_matrix[dem_indices]  # type: ignore
    rep_user_term_matrix: sp.csr_matrix = user_term_matrix[rep_indices]  # type: ignore

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

        random_users_dem, random_users_rep = random_shuffle_users(user_term_matrix)

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


def get_balanced_partisan_user_tokens(
    dem_user_tokens: pd.DataFrame,
    rep_user_tokens: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dem_user_cnt: int = dem_user_tokens.shape[0]
    rep_user_cnt: int = rep_user_tokens.shape[0]

    if dem_user_cnt < 10 or rep_user_cnt < 10:
        # return these values when there is not enough data to make predictions on
        raise RuntimeWarning(
            f"""Not enough data to make predictions:
            - Number of democrat users: {dem_user_cnt}
            - Number of republican users: {rep_user_cnt}
            """
        )

    RNG = np.random.default_rng(SEED)

    # make the prior neutral (i.e. make sure there are the same number of dem and rep users)
    if dem_user_cnt > rep_user_cnt:
        random_ind = RNG.choice(dem_user_cnt, replace=False, size=rep_user_cnt)
        sample_dem_user_tokens = dem_user_tokens.iloc[random_ind]
        return sample_dem_user_tokens, rep_user_tokens

    elif rep_user_cnt > dem_user_cnt:
        random_ind = RNG.choice(rep_user_cnt, replace=False, size=dem_user_cnt)
        sample_rep_user_tokens = rep_user_tokens.iloc[random_ind]
        return dem_user_tokens, sample_rep_user_tokens
    else:
        return dem_user_tokens, rep_user_tokens


def random_shuffle_users(user_term_matrix: sp.csr_matrix):
    RNG = np.random.default_rng(SEED)

    user_cnt: int = user_term_matrix.shape[0]

    shuffled_user_ind = RNG.choice(user_cnt, replace=False, size=user_cnt)

    random_users_dem: sp.csr_matrix = user_term_matrix[
        shuffled_user_ind[: int(user_cnt / 2)]
    ]  # type: ignore
    random_users_rep: sp.csr_matrix = user_term_matrix[
        shuffled_user_ind[int(user_cnt / 2) :]
    ]  # type: ignore

    return random_users_dem, random_users_rep


def calculate_polarization_by_time(
    event_comments,
    ngram_range: Tuple[int, int],
    event_vocab: Dict[str, int],
    freq="D",
    equalize_users: bool = True,
):
    polarization = []
    for datetime, date_comments in event_comments.groupby(
        pd.Grouper(key="datetime", freq=freq)
    ):
        print(datetime)
        (total_polarization, random_polarization, user_cnt), _ = calculate_polarization(
            comments=date_comments,
            ngram_range=ngram_range,
            event_vocab=event_vocab,
            method="leaveout",
            equalize_users=equalize_users,
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
