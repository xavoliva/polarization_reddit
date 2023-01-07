import math
from collections import defaultdict


def logodds_with_prior(
    vocab: dict[str, int],
    dem_vocab: dict[str, int],
    rep_vocab: dict[str, int],
) -> dict[str, float]:
    """
    Weighted log odds ratio, as defined in
    https://bookdown.org/Maxine/tidy-text-mining/weighted-log-odds-ratio.html
    """

    nr_tokens = sum(vocab.values())
    nr_dem_tokens = sum(dem_vocab.values())
    nr_rep_tokens = sum(rep_vocab.values())

    logodds = defaultdict(float)

    for word in vocab.keys():
        dem_numerator = dem_vocab[word] + vocab[word]
        dem_denominator = nr_dem_tokens + nr_tokens - dem_numerator

        rep_numerator = rep_vocab[word] + vocab[word]
        rep_denominator = nr_rep_tokens + nr_tokens - rep_numerator

        raw_logodds = math.log(dem_numerator / dem_denominator) - math.log(
        rep_numerator / rep_denominator
        )

        variance = (1 / dem_numerator) + (1 / rep_numerator)

        logodds[word] = raw_logodds / math.sqrt(variance)

    return logodds
