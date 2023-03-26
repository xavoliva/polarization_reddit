"""
Pre-processing constants
"""
import pandas as pd

from load.constants import DATA_DIR

EVENTS_DIR = f"{DATA_DIR}/events"
OUTPUT_DIR = f"{DATA_DIR}/output"
METADATA_DIR = f"{DATA_DIR}/metadata"

from nltk.stem.lancaster import LancasterStemmer

sno = LancasterStemmer()


def get_event_regex(general_keywords, event_keywords, operator):
    if operator == "or":
        regex = "|".join(
            [
                sno.stem(word)
                for keyword in (general_keywords + event_keywords)
                for word in keyword.split()
            ]
        )
    elif operator == "and":
        regex = (
            f"({'|'.join([sno.stem(word) for keyword in general_keywords for word in keyword.split()])})"
            + "&"
            + f"({'|'.join([sno.stem(word) for keyword in event_keywords for word in keyword.split()])})"
        )

    return regex


# ELECTIONS

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


ELECTION_EVENTS = ELECTIONS_EVENTS_INFO.keys()

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
            ELECTIONS_KEYWORDS,
            [
                "charleston",
            ],
            "and",
        ),
    },
    "chattanooga_shooting": {
        "name": "Chattanooga shooting",
        "date": pd.to_datetime("07-16-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "chattanooga",
            ],
            "and",
        ),
    },
    "roseburg_shooting": {
        "name": "Roseburg shooting",
        "date": pd.to_datetime("10-01-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "umpqua",
                "roseburg",
            ],
            "and",
        ),
    },
    "colorado_springs_shooting": {
        "name": "Colorado Springs shooting",
        "date": pd.to_datetime("11-27-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "colorado springs",
                "planned parenthood",
            ],
            "and",
        ),
    },
    "san_bernardino_shooting": {
        "name": "San Bernardino shooting",
        "date": pd.to_datetime("12-02-2015"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "san bernardino",
            ],
            "and",
        ),
    },
    "kalamazoo_shooting": {
        "name": "Kalamazoo shooting",
        "date": pd.to_datetime("02-20-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "kalamazoo",
            ],
            "and",
        ),
    },
    "orlando_nightclub_shooting": {
        "name": "Orlando nightclub shooting",
        "date": pd.to_datetime("06-12-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "orlando",
                "pulse nightclub",
            ],
            "and",
        ),
    },
    "dallas_police_shooting": {
        "name": "Dallas police shooting",
        "date": pd.to_datetime("07-07-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "dallas",
            ],
            "and",
        ),
    },
    "baton_rouge_police_shooting": {
        "name": "Baton Rouge police shooting",
        "date": pd.to_datetime("07-17-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "baton rouge",
            ],
            "and",
        ),
    },
    "burlington_shooting": {
        "name": "Burlington shooting",
        "date": pd.to_datetime("09-16-2016"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "colorado springs",
                "planned parenthood",
            ],
            "and",
        ),
        "keywords": {
            "burlington",
            "cascade mall",
        },
    },
    "fort_lauderdale_airport_shooting": {
        "name": "Fort Lauderdale airport shooting",
        "date": pd.to_datetime("01-06-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "fort lauderdale",
            ],
            "and",
        ),
    },
    "fresno_shooting": {
        "name": "Fresno shooting",
        "date": pd.to_datetime("04-18-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "fresno",
            ],
            "and",
        ),
    },
    "san_francisco_cafe_shooting": {
        "name": "San Francisco cafe shooting",
        "date": pd.to_datetime("06-14-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "ups",
                "san francisco",
            ],
            "and",
        ),
    },
    "vegas_shooting": {
        "name": "Las Vegas shooting",
        "date": pd.to_datetime("10-01-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "vegas",
                "harvest festival",
                "mandalay bay",
                "route 91",
            ],
            "and",
        ),
    },
    "thornton_walmart_shooting": {
        "name": "Thornton Walmart shooting",
        "date": pd.to_datetime("11-01-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "thornton",
                "walmart",
                "denver",
            ],
            "and",
        ),
    },
    "sutherland_springs_church_shooting": {
        "name": "Sutherland Springs church shooting",
        "date": pd.to_datetime("11-05-2017"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "sutherland springs",
            ],
            "and",
        ),
    },
    "parkland_school_shooting": {
        "name": "Parkland school shooting",
        "date": pd.to_datetime("02-14-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "parkland",
                "marjory stoneman",
            ],
            "and",
        ),
    },
    "nashville_waffle_house_shooting": {
        "name": "Nashville Waffle House shooting",
        "date": pd.to_datetime("04-22-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "nashville",
                "waffle house",
            ],
            "and",
        ),
    },
    "santa_fe_high_school_shooting": {
        "name": "Santa Fe High School shooting",
        "date": pd.to_datetime("05-18-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "santa fe",
            ],
            "and",
        ),
    },
    "annapolis_journal_shooting": {
        "name": "Annapolis Journal shooting",
        "date": pd.to_datetime("06-28-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "annapolis",
                "capital gazette",
            ],
            "and",
        ),
    },
    "pittsburgh_synagogue_shooting": {
        "name": "Pittsburgh synagogue shooting",
        "date": pd.to_datetime("10-27-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "pittsburgh",
                "tree of life",
            ],
            "and",
        ),
    },
    "thousand_oaks_bar_shooting": {
        "name": "Thousand Oaks bar shooting",
        "date": pd.to_datetime("11-07-2018"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "thousand oaks",
            ],
            "and",
        ),
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

# TBD
