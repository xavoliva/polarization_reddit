"""
Event constants
"""
import pandas as pd

DATA_DIR = "data"
INPUT_DIR = f"{DATA_DIR}/input"
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

EVENTS = EVENTS_INFO.keys()
