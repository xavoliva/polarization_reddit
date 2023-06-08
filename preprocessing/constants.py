"""
Pre-processing constants
"""
from typing import List

import pandas as pd


from load.constants import DATA_DIR

EVENTS_DIR = f"{DATA_DIR}/events"
OUTPUT_DIR = f"{DATA_DIR}/output"
METADATA_DIR = f"{DATA_DIR}/metadata"

MIN_OCCURENCE_FOR_VOCAB = 50

from nltk.stem.lancaster import LancasterStemmer

sno = LancasterStemmer()


def get_event_regex(general_keywords, event_keywords, operator):
    if operator == "or":
        regex = r"\b(?:"

        for keyword in general_keywords + event_keywords:
            if " " in keyword:
                regex += rf"{' '.join([sno.stem(word) for word in keyword.split()])}|"
            else:
                regex += rf"{sno.stem(keyword)}|"

        regex = regex[:-1] + r")\b"

    elif operator == "and":
        regex = r"\b(?:"

        for keyword in general_keywords:
            if " " in keyword:
                regex += rf"{' '.join([sno.stem(word) for word in keyword.split()])}|"
            else:
                regex += rf"{sno.stem(keyword)}|"

        regex = rf"{regex[:-1]})\b.*\b(?:"

        for keyword in event_keywords:
            if " " in keyword:
                regex += rf"{' '.join([sno.stem(word) for word in keyword.split()])}|"
            else:
                regex += rf"{sno.stem(keyword)}|"

        regex = rf"{regex[:-1]})\b"

    else:
        raise ValueError("Operator must be 'or' or 'and'")

    return regex


# ELECTIONS

ELECTIONS_KEYWORDS = [
    "vote",
    "president",
    "candidate",
    "nominee",
    # "democrat",
    # "republican",
    "election",
    "ballot",
    "swing state",
    # "election poll",
    # "primaries",
    "campaign",
    # "midterm",
    # "governor",
]

ELECTIONS_EVENTS_INFO = {
    "us_elections_2008": {
        "name": "2008 US presidential election",
        "date": pd.to_datetime("11-04-2008"),
        "relevant_dates": {},
        "type": "election",
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS, ["obama", "mccain", "biden", "palin"], "or"
        ),
    },
    "us_elections_2012": {
        "name": "2012 US presidential election",
        "date": pd.to_datetime("11-06-2012"),
        "relevant_dates": {},
        "type": "election",
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "obama",
                "biden",
                "romney",
                "paul ryan",
            ],
            "or",
        ),
    },
    "us_midterms_2014": {
        "name": "2014 US midterm elections",
        "date": pd.to_datetime("11-04-2014"),
        "relevant_dates": {},
        "type": "election",
        "regex": get_event_regex(ELECTIONS_KEYWORDS, [], "or"),
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
        "type": "election",
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "trump",
                "pence",
                "clinton",
                "tim kaine",
                "bernie sanders",
                "ted cruz",
                "jeb bush",
                "john kasich",
                "marco rubio",
                "carly fiorina",
            ],
            "or",
        ),
    },
    "us_midterms_2018": {
        "name": "2018 US midterm elections",
        "date": pd.to_datetime("11-06-2018"),
        "relevant_dates": {},
        "type": "election",
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [],
            "or",
        ),
    },
}


ELECTION_EVENTS = list(ELECTIONS_EVENTS_INFO)

# MASS SHOOTINGS

MASS_SHOOTINGS_KEYWORDS = [
    "shoot",
    "gun",
    "kill",
    "attack",
    "massacre",
    "victim",
]

MASS_SHOOTINGS_EVENTS_INFO = {
    "charleston_church_shooting": {
        "name": "Charleston church shooting",
        "date": pd.to_datetime("06-17-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "charleston",
            ],
            "and",
        ),
        "keywords": [
            "charleston",
        ],
    },
    "chattanooga_shooting": {
        "name": "Chattanooga shooting",
        "date": pd.to_datetime("07-16-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "chattanooga",
            ],
            "and",
        ),
        "keywords": [
            "chattanooga",
        ],
    },
    "roseburg_shooting": {
        "name": "Roseburg shooting",
        "date": pd.to_datetime("10-01-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "umpqua",
                "roseburg",
            ],
            "and",
        ),
        "keywords": [
            "umpqua",
            "roseburg",
        ],
    },
    "colorado_springs_shooting": {
        "name": "Colorado Springs shooting",
        "date": pd.to_datetime("11-27-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "colorado springs",
                "planned parenthood",
            ],
            "and",
        ),
        "keywords": [
            "colorado springs",
            "planned parenthood",
        ],
    },
    "san_bernardino_shooting": {
        "name": "San Bernardino shooting",
        "date": pd.to_datetime("12-02-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "san bernardino",
            ],
            "and",
        ),
        "keywords": [
            "san bernardino",
        ],
    },
    "kalamazoo_shooting": {
        "name": "Kalamazoo shooting",
        "date": pd.to_datetime("02-20-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "kalamazoo",
            ],
            "and",
        ),
        "keywords": [
            "kalamazoo",
        ],
    },
    "orlando_nightclub_shooting": {
        "name": "Orlando nightclub shooting",
        "date": pd.to_datetime("06-12-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "orlando",
                "pulse nightclub",
            ],
            "and",
        ),
        "keywords": [
            "orlando",
            "pulse nightclub",
        ],
    },
    "dallas_police_shooting": {
        "name": "Dallas police shooting",
        "date": pd.to_datetime("07-07-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "dallas",
            ],
            "and",
        ),
        "keywords": [
            "dallas",
        ],
    },
    "baton_rouge_police_shooting": {
        "name": "Baton Rouge police shooting",
        "date": pd.to_datetime("07-17-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "baton rouge",
            ],
            "and",
        ),
        "keywords": [
            "baton rouge",
        ],
    },
    "burlington_shooting": {
        "name": "Burlington shooting",
        "date": pd.to_datetime("09-16-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "burlington",
                "cascade mall",
            ],
            "and",
        ),
        "keywords": [
            "burlington",
            "cascade mall",
        ],
    },
    "fort_lauderdale_airport_shooting": {
        "name": "Fort Lauderdale airport shooting",
        "date": pd.to_datetime("01-06-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "fort lauderdale",
            ],
            "and",
        ),
        "keywords": [
            "fort lauderdale",
        ],
    },
    "fresno_shooting": {
        "name": "Fresno shooting",
        "date": pd.to_datetime("04-18-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "fresno",
            ],
            "and",
        ),
        "keywords": [
            "fresno",
        ],
    },
    "san_francisco_cafe_shooting": {
        "name": "San Francisco cafe shooting",
        "date": pd.to_datetime("06-14-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "ups",
                "san francisco",
            ],
            "and",
        ),
        "keywords": [
            "ups",
            "san francisco",
        ],
    },
    "vegas_shooting": {
        "name": "Las Vegas shooting",
        "date": pd.to_datetime("10-01-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "vegas",
                "harvest festival",
                "mandalay bay",
                "route 91",
            ],
            "and",
        ),
        "keywords": [
            "vegas",
            "harvest festival",
            "mandalay bay",
            "route 91",
        ],
    },
    "thornton_walmart_shooting": {
        "name": "Thornton Walmart shooting",
        "date": pd.to_datetime("11-01-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "thornton",
                "walmart",
                "denver",
            ],
            "and",
        ),
        "keywords": [
            "thornton",
            "walmart",
            "denver",
        ],
    },
    "sutherland_springs_church_shooting": {
        "name": "Sutherland Springs church shooting",
        "date": pd.to_datetime("11-05-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "sutherland springs",
            ],
            "and",
        ),
        "keywords": [
            "sutherland springs",
        ],
    },
    "parkland_school_shooting": {
        "name": "Parkland school shooting",
        "date": pd.to_datetime("02-14-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "parkland",
                "marjory stoneman",
            ],
            "and",
        ),
        "keywords": [
            "parkland",
            "marjory stoneman",
        ],
    },
    "nashville_waffle_house_shooting": {
        "name": "Nashville Waffle House shooting",
        "date": pd.to_datetime("04-22-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "nashville",
                "waffle house",
            ],
            "and",
        ),
        "keywords": [
            "nashville",
            "waffle house",
        ],
    },
    "santa_fe_high_school_shooting": {
        "name": "Santa Fe High School shooting",
        "date": pd.to_datetime("05-18-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "santa fe",
            ],
            "and",
        ),
        "keywords": [
            "santa fe",
        ],
    },
    "annapolis_journal_shooting": {
        "name": "Annapolis Journal shooting",
        "date": pd.to_datetime("06-28-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "annapolis",
                "capital gazette",
            ],
            "and",
        ),
        "keywords": [
            "annapolis",
            "capital gazette",
        ],
    },
    "pittsburgh_synagogue_shooting": {
        "name": "Pittsburgh synagogue shooting",
        "date": pd.to_datetime("10-27-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "pittsburgh",
                "tree of life",
            ],
            "and",
        ),
        "keywords": [
            "pittsburgh",
            "tree of life",
        ],
    },
    "thousand_oaks_bar_shooting": {
        "name": "Thousand Oaks bar shooting",
        "date": pd.to_datetime("11-07-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                "thousand oaks",
            ],
            "and",
        ),
        "keywords": [
            "thousand oaks",
        ],
    },
    # "bakersfield_shooting": {
    #     "name": "Bakersfield shooting",
    #     "date": pd.to_datetime("11-24-2018"),
    #     "relevant_dates": {},
    # },
    # "chicago_bar_shooting": {
    #     "name": "Chicago bar shooting",
    #     "date": pd.to_datetime("02-14-2019"),
    #     "relevant_dates": {},
    # },
    # "el_paso_shooting": {
    #     "name": "El Paso shooting",
    #     "date": pd.to_datetime("08-03-2019"),
    #     "relevant_dates": {},
    # },
    # "dayton_shooting": {
    #     "name": "Dayton shooting",
    #     "date": pd.to_datetime("08-04-2019"),
    #     "relevant_dates": {},
    # },
}

# ABORTION
# https://19thnews.org/2021/12/abortion-in-america-photos-visual-timeline/

# https://en.wikipedia.org/wiki/Abortion_debate
ABORTION_KEYWORDS = [
    "abortion",
    "planned parenthood",
    "roe vs",
    "vs wade",
    "pro choice",
    "pro life",
    "anti choice",
    "anti life",
    "infanticide",
]


ABORTION_EVENTS_INFO = {
    "dr_george_tiller_shooting": {
        "name": "Dr. George Tiller shooting",
        "date": pd.to_datetime("05-31-2009"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "north_dakota_heartbeat_bill": {
        "name": "North Dakota heartbeat bill",
        "date": pd.to_datetime("03-26-2013"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "texas_house_bill_2": {
        "name": "Texas House Bill 2",
        "date": pd.to_datetime("07-18-2013"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "colorado_springs_shooting": {
        "name": "Colorado Springs shooting",
        "date": pd.to_datetime("11-27-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "womans_health_vs_hellerstedt": {
        "name": "Whole Woman's Health v. Hellerstedt",
        "date": pd.to_datetime("06-27-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "ohio_heartbeat_bill": {
        "name": "Ohio heartbeat bill",
        "date": pd.to_datetime("11-04-2019"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "georgia_heartbeat_bill": {
        "name": "Georgia heartbeat bill",
        "date": pd.to_datetime("04-04-2019"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
}

# TBD
