"""
Constants for project
"""
import random

import pandas as pd

RNG = random.Random()
RNG.seed(42)

DATA_DIR = "data"
INPUT_DIR = f"{DATA_DIR}/input"
EVENTS_DIR = f"{DATA_DIR}/events"
OUTPUT_DIR = f"{DATA_DIR}/output"
FIGURES_DIR = f"{DATA_DIR}/figures"

EVENTS = ["brexit", "us_election"]

EVENTS_INFO = {
    "brexit": {"name": "Brexit referendum", "date": pd.to_datetime("06-23-2016")},
    "us_election": {
        "name": "US presidential election",
        "date": pd.to_datetime("11-08-2016"),
    },
}

ALL_COLUMNS = [
    "archived",
    "author",
    "author_flair_css_class",
    "author_flair_text",
    "body",
    "controversiality",
    "created_utc",
    "distinguished",
    "downs",
    "edited",
    "gilded",
    "id",
    "link_id",
    "name",
    "parent_id",
    "removal_reason",
    "retrieved_on",
    "score",
    "score_hidden",
    "subreddit",
    "subreddit_id",
    "ups",
]

COLUMNS = ["author", "body", "subreddit", "subreddit_id", "created_utc"]

DEM_SUBREDDITS = set(
    [
        "democrats",
        "AOC",
        "Anarchism",
        "AnarchismOnline",
        "AnarchistNews",
        "BadSocialScience",
        "DemocraticSocialism",
        "socialism",
        "Socialism_101",
        "AntiTrumpAlliance",
        "BernieSanders",
        "DemocraticSocialism",
        "ENLIGHTENEDCENTRISM",
        "ShitLiberalsSay",
        "Socialism_101",
        "antifa",
        "chomsky",
        "occupywallstreet",
        "progressive",
        "rojava",
        "AntiTrumpAlliance",
        "FULLCOMMUNISM",
        "GreenAndPleasant",
        "HillaryForAmerica",
        "IWW",
        "Impeach_Trump",
        "Sino",
        "TrumpForPrison",
    ]
)

REP_SUBREDDITS = set(
    [
        "republicans",
        "Anarcho_Capitalism",
        "Conservative",
        "ConservativeLounge",
        "ConservativeMeta",
        "DrainTheSwamp",
        "ShitPoliticsSays",
        "Republican",
        "RepublicanValues",
        "progun",
        "antifapassdenied",
        "Capitalism",
        "Libertarian",
        "BernieSandersSucks",
        "Identitarians",
        "NationalSocialism",
        "monarchism",
        "neoliberal",
        "pol",
        "AltRightChristian",
        "HillaryForPrison",
        "JordanPeterson",
        "altright",
    ]
)

PARTISAN_SUBREDDITS = DEM_SUBREDDITS | REP_SUBREDDITS

CEN_SUBREDDITS = ["worldnews", "politics", "news"]

MIN_OCCURENCE_FOR_VOCAB = 25
