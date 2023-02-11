from typing import Dict

import numpy as np


def logodds_with_prior(
    term_vec,
    dem_term_vec,
    rep_term_vec,
) -> Dict[str, float]:
    """
    Weighted log odds ratio, as defined in
    https://bookdown.org/Maxine/tidy-text-mining/weighted-log-odds-ratio.html
    """
    term_vec = term_vec.toarray()[0]
    dem_term_vec = dem_term_vec.toarray()[0]
    rep_term_vec = rep_term_vec.toarray()[0]

    nr_tokens = np.sum(term_vec)
    nr_dem_tokens = np.sum(dem_term_vec)
    nr_rep_tokens = np.sum(rep_term_vec)

    dem_numerator = dem_term_vec + term_vec
    dem_denominator = nr_dem_tokens + nr_tokens - dem_numerator

    rep_numerator = rep_term_vec + term_vec
    rep_denominator = nr_rep_tokens + nr_tokens - rep_numerator

    raw_logodds = np.log(dem_numerator / dem_denominator) - np.log(
        rep_numerator / rep_denominator
    )

    variance = 1 / dem_numerator + 1 / rep_numerator

    logodds = raw_logodds / np.sqrt(variance)

    return logodds
