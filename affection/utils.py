from typing import Dict
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_compound_sentiment_score(comment: str) -> Dict[str, float]:
    # Create a SentimentIntensityAnalyzer object
    sia = SentimentIntensityAnalyzer()

    # Polarity scores are floats within the range [-1.0, 1.0]
    polarity_scores = sia.polarity_scores(comment)["compound"]

    return polarity_scores


def get_comments_mentioning_opposition(comments, body, party, regexes):
    if party == "rep":
        party_regex = regexes["rep"]
        opposition_regex = regexes["dem"]
        type = "rep_comments_about_dems"
    else:
        party_regex = regexes["dem"]
        opposition_regex = regexes["rep"]
        type = "dem_comments_about_reps"
    opposition_comments = comments[
        (comments["party"] == party)
        & (
            comments[body].str.contains(opposition_regex, regex=True)
            & ~comments[body].str.contains(party_regex, regex=True)
        )
    ].copy()

    opposition_comments["type"] = type

    return opposition_comments
