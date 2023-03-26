"""
Pre-processing constants
"""
import pandas as pd

from load.constants import DATA_DIR

EVENTS_DIR = f"{DATA_DIR}/events"
OUTPUT_DIR = f"{DATA_DIR}/output"
METADATA_DIR = f"{DATA_DIR}/metadata"

ELECTIONS_EVENTS_INFO = {
    "us_elections_2008": {
        "name": "2008 US presidential election",
        "date": pd.to_datetime("11-04-2008"),
        "relevant_dates": {},
    },
    "us_elections_2012": {
        "name": "2012 US presidential election",
        "date": pd.to_datetime("11-06-2012"),
        "relevant_dates": {},
    },
    "us_midterms_2014": {
        "name": "2014 US midterm elections",
        "date": pd.to_datetime("11-04-2014"),
        "relevant_dates": {},
    },
    "us_elections_2016": {
        "name": "2016 US presidential election",
        "date": pd.to_datetime("11-08-2016"),
        "relevant_dates": {
            "Super Tuesday": pd.to_datetime("03-01-2016"),
            "Donald Trump secures Republican presidential nomination": pd.to_datetime(
                "05-26-2016"
            ),
            "Hillary Clinton secures Democratic presidential nomination": pd.to_datetime(
                "06-06-2016"
            ),
            "First presidential general election debate": pd.to_datetime("09-26-2016"),
            "Leaked tape & WikiLeaks publication": pd.to_datetime("10-07-2016"),
            "Second presidential general election debate": pd.to_datetime("10-09-2016"),
            "Third presidential general election debate": pd.to_datetime("10-19-2016"),
        },
    },
    "us_midterms_2018": {
        "name": "2018 US midterm elections",
        "date": pd.to_datetime("11-06-2018"),
        "relevant_dates": {},
    },
}

ELECTIONS_KEYWORDS = [
    "vote",
    "president",
    "candidate",
    "nominee",
    "democrat",
    "republican",
    "election",
    "ballot",
    "swing state",
    "swing district",
    "swing county",
    "swing voter",
    "poll",
    "primaries",
    "campaign",
    "midterm",
    "governor",
]

ELECTIONS_KEYWORDS_2008 = [
    "obama",
    "mccain",
    "biden",
    "palin",
]

ELECTIONS_KEYWORDS_2012 = [
    "obama",
    "biden",
    "romney",
    "paul ryan",
]

ELECTIONS_KEYWORDS_2016 = [
    "trump",
    "pence",
    "clinton",
    "tim kaine" "bernie sanders",
    "ted cruz",
    "jeb bush",
]

from nltk.stem.lancaster import LancasterStemmer

sno = LancasterStemmer()

ELECTIONS_REGEX = {
    2008: "|".join(
        [sno.stem(token) for token in ELECTIONS_KEYWORDS + ELECTIONS_KEYWORDS_2008]
    ),
    2012: "|".join(
        [sno.stem(token) for token in ELECTIONS_KEYWORDS + ELECTIONS_KEYWORDS_2012]
    ),
    2016: "|".join(
        [sno.stem(token) for token in ELECTIONS_KEYWORDS + ELECTIONS_KEYWORDS_2016]
    ),
}

EVENTS = ELECTIONS_EVENTS_INFO.keys()

MIN_OCCURENCE_FOR_VOCAB = 25
