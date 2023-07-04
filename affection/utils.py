from typing import Dict
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_compound_sentiment_score(comment: str) -> Dict[str, float]:
    # Create a SentimentIntensityAnalyzer object
    sia = SentimentIntensityAnalyzer()

    # Polarity scores are floats within the range [-1.0, 1.0]
    polarity_scores = sia.polarity_scores(comment)["compound"]

    return polarity_scores