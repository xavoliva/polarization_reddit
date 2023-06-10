from typing import Dict

import numpy as np


def logodds_with_prior(
    term_vec,
    dem_term_vec,
    rep_term_vec,
    zscore=True,
):
    """
    Weighted log odds ratio with uninformative Dirichlet prior, as defined in
    https://bookdown.org/Maxine/tidy-text-mining/weighted-log-odds-ratio.html
    """
    n_dem = np.sum(dem_term_vec)
    n_rep = np.sum(rep_term_vec)


    alpha_0 = np.sum(term_vec)

    has_zero = np.any(term_vec == 0)

    # check validitiy of input
    assert not has_zero
    assert alpha_0 == n_dem + n_rep

    alpha_w = term_vec
    y_w_dem = dem_term_vec
    y_w_rep = rep_term_vec

    omega_w_dem = (y_w_dem + alpha_w) / (
        (n_dem + alpha_0) - (y_w_dem + alpha_w)
    )  # odds in group dem
    omega_w_rep = (y_w_rep + alpha_w) / (
        (n_rep + alpha_0) - (y_w_rep + alpha_w)
    )  # odds in group rep

    delta_w_dem_rep = np.log(omega_w_dem) - np.log(omega_w_rep)  # eqn 16

    if zscore:
        sigma2_w_dem_rep = 1 / (y_w_dem + alpha_w) + 1 / (y_w_rep + alpha_w)  # eqn 20
        zeta_w_dem_rem = delta_w_dem_rep / np.sqrt(sigma2_w_dem_rep)   # eqn 22
        return zeta_w_dem_rem

    return delta_w_dem_rep
