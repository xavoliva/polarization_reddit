import pandas as pd

from events.utils import get_event_regex

ELECTIONS_KEYWORDS = [
    "vote",
    "president",
    "candidate",
    "nominee",
    "election",
    "ballot",
    "swing state",
    "campaign",
]

ELECTIONS_EVENTS_INFO = {
    "us_elections_2012": {
        "name": "2012 US presidential election",
        "date": pd.to_datetime("2012-11-06"),
        "relevant_dates": {
            "Super Tuesday": pd.to_datetime("2012-03-06"),
            "Barack Obama Democratic presidential nomination": pd.to_datetime(
                "2012-09-06"
            ),
            "Mitt Romney Republican presidential nomination": pd.to_datetime(
                "2012-08-30"
            ),
            "First presidential election debate": pd.to_datetime("2012-10-03"),
        },
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
        "date": pd.to_datetime("2014-11-04"),
        "relevant_dates": {},
        "type": "election",
        "regex": get_event_regex(ELECTIONS_KEYWORDS, ["midterms"], "or"),
    },
    "us_elections_2016": {
        "name": "2016 US presidential election",
        "date": pd.to_datetime("2016-11-08"),
        "relevant_dates": {
            "Super Tuesday": pd.to_datetime("2016-03-01"),
            "Donald Trump Republican presidential nomination": pd.to_datetime(
                "2016-07-21"
            ),
            "Hillary Clinton Democratic presidential nomination": pd.to_datetime(
                "2016-07-28"
            ),
            "First presidential election debate": pd.to_datetime("2016-09-26"),
            "Leaked tape & WikiLeaks publication": pd.to_datetime("2016-10-07"),
            # "Second presidential election debate": pd.to_datetime("2016-10-09"),
            # "Third presidential election debate": pd.to_datetime("2016-10-19"),
        },
        "type": "election",
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            [
                "trump",
                "pence",
                "clinton",
                "tim kaine",
                "bernie",
                "sanders",
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
        "date": pd.to_datetime("2018-11-06"),
        "relevant_dates": {},
        "type": "election",
        "regex": get_event_regex(
            ELECTIONS_KEYWORDS,
            ["midterms"],
            "or",
        ),
    },
}

ELECTION_EVENTS = list(ELECTIONS_EVENTS_INFO)
