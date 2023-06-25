import pandas as pd

from events.utils import get_event_regex

MASS_SHOOTINGS_KEYWORDS = [
    "shoot",
    "gun",
    "kill",
    "attack",
    "massacre",
    "victim",
]

GUN_CONTROL_SUBREDDITS = [
    "2ALiberals",
    "AsAGunOwner",
    "GunsAreCool",
    "NOWTTYG",
    "actualliberalgunowner",
    "guncontrol",
    "gunpolitics",
    "liberalgunowners",
    "neveragainmovement",
    "nra",
    "progun",
    "secondamendment",
    "shitguncontrollerssay",
]

MASS_SHOOTINGS_EVENTS_INFO = {
    "charleston_church_shooting": {
        "name": "Charleston church shooting",
        "date": pd.to_datetime("2015-06-17"),
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
        "date": pd.to_datetime("2015-07-16"),
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
        "date": pd.to_datetime("2015-10-01"),
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
        "date": pd.to_datetime("2015-11-27"),
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
        "date": pd.to_datetime("2015-12-02"),
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
        "date": pd.to_datetime("2016-02-20"),
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
        "date": pd.to_datetime("2016-06-12"),
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
        "date": pd.to_datetime("2016-07-07"),
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
        "date": pd.to_datetime("2016-07-17"),
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
        "date": pd.to_datetime("2016-09-16"),
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
        "date": pd.to_datetime("2017-01-06"),
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
        "date": pd.to_datetime("2017-04-18"),
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
        "date": pd.to_datetime("2017-06-14"),
        "relevant_dates": {},
        "regex": get_event_regex(
            MASS_SHOOTINGS_KEYWORDS,
            [
                # "ups",
                "san francisco",
            ],
            "and",
        ),
        "keywords": [
            # "ups",
            "san francisco",
        ],
    },
    "vegas_shooting": {
        "name": "Las Vegas shooting",
        "date": pd.to_datetime("2017-10-01"),
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
        "date": pd.to_datetime("2017-11-01"),
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
        "date": pd.to_datetime("2017-11-05"),
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
        "date": pd.to_datetime("2018-02-14"),
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
        "date": pd.to_datetime("2018-04-22"),
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
        "date": pd.to_datetime("2018-05-18"),
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
        "date": pd.to_datetime("2018-06-28"),
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
        "date": pd.to_datetime("2018-10-27"),
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
        "date": pd.to_datetime("2018-11-07"),
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
}
