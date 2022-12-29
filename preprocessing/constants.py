"""
Pre-processing constants
"""
import pandas as pd

DATA_DIR = "data"
EVENTS_DIR = f"{DATA_DIR}/events"
OUTPUT_DIR = f"{DATA_DIR}/output"
FIGURES_DIR = f"{DATA_DIR}/figures"

EVENTS_INFO = {
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

ELECTIONS_REGEX = "vote|us|election|trump|hillary|president|candidate|democrat|republican|donald|clinton|reps|dems|elect|ballot|crooked|swing"

EVENTS = EVENTS_INFO.keys()

MIN_OCCURENCE_FOR_VOCAB = 25
