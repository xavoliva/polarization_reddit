"""
Pre-processing constants
"""
import pandas as pd

from load.constants import DATA_DIR

EVENTS_DIR = f"{DATA_DIR}/events"
OUTPUT_DIR = f"{DATA_DIR}/output"

EVENTS_INFO = {
    "us_elections_2008": {
        "name": "2008 US presidential election",
        "date": pd.to_datetime("11-04-2008"),
    },
    "us_elections_2012": {
        "name": "2012 US presidential election",
        "date": pd.to_datetime("11-06-2012"),
    },
    "us_midterms_2014": {
        "name": "2014 US midterm elections",
        "date": pd.to_datetime("11-04-2014"),
    },
    "us_elections_2016": {
        "name": "2016 US presidential election",
        "date": pd.to_datetime("11-08-2016"),
    },
    "us_midterms_2018": {
        "name": "2018 US midterm elections",
        "date": pd.to_datetime("11-06-2018"),
    },
}

ELECTIONS_KEYWORDS = [
    "vote",
    "us",
    "usa",
    "election",
    "president",
    "candidate",
    "democrat",
    "republican",
    "rep",
    "dem",
    "elect",
    "ballot",
    "swing",
]

ELECTIONS_KEYWORDS_2008 = [
    "barack",
    "obama",
    "john mccain",
    "mccain",
    "joe",
    "biden",
    "sarah palin",
    "palin",
]

ELECTIONS_KEYWORDS_2012 = [
    "barack",
    "obama",
    "mitt romney",
    "romney",
    "paul ryan",
    "ryan",
]

ELECTIONS_KEYWORDS_2016 = [
    "donald",
    "trump",
    "hillary",
    "clinton",
]


ELECTIONS_REGEX = {
    2008: "|".join(ELECTIONS_KEYWORDS + ELECTIONS_KEYWORDS_2008),
    2012: "|".join(ELECTIONS_KEYWORDS + ELECTIONS_KEYWORDS_2012),
    2016: "|".join(ELECTIONS_KEYWORDS + ELECTIONS_KEYWORDS_2016),
}

EVENTS = EVENTS_INFO.keys()

MIN_OCCURENCE_FOR_VOCAB = 25
