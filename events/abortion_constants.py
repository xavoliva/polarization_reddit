import pandas as pd

from events.utils import get_event_regex

# ABORTION
# https://19thnews.org/2021/12/abortion-in-america-photos-visual-timeline/

# https://en.wikipedia.org/wiki/Abortion_debate
ABORTION_KEYWORDS = [
    "abortion",
    "fetus",
    "pregnancy",
    "unborn",
    "birth control",
    "planned parenthood",
    "roe vs wade",
    "health v hellerstedt",
    "heartbeat bill",
    "pro choice",
    "pro life",
    "anti choice",
    "anti life",
    "infanticide",
]

# TODO: ADD RELEVANT EVENT KEYWORDS
ABORTION_EVENTS_INFO = {
    "abortion": {
        "name": "Abortion",
        "relevant_dates": {
            "Colorado Springs shooting": pd.to_datetime("2015-11-27"),
            "Whole Woman's Health v. Hellerstedt": pd.to_datetime("2016-06-27"),
            "Ohio heartbeat bill": pd.to_datetime("2019-11-04"),
            "Georgia heartbeat bill": pd.to_datetime("2019-04-04"),
        },
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),

    },
    "colorado_springs_shooting": {
        "name": "Colorado Springs shooting",
        "date": pd.to_datetime("2015-11-27"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ABORTION_KEYWORDS,
            [
                "colorado springs",
                "shooting",
            ],
            "and",
        ),
    },
    "womans_health_vs_hellerstedt": {
        "name": "Whole Woman's Health v. Hellerstedt",
        "date": pd.to_datetime("2016-06-27"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ABORTION_KEYWORDS,
            [
                "supreme court",
                "ruling",
                "lawsuit",
                "health v hellerstedt",
            ],
            "and",
        ),
    },
    "ohio_heartbeat_bill": {
        "name": "Ohio heartbeat bill",
        "date": pd.to_datetime("2019-11-04"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ABORTION_KEYWORDS,
            [
                "ohio",
                "heartbeat bill",
                "six week",
            ],
            "and",
        ),
    },
    "georgia_heartbeat_bill": {
        "name": "Georgia heartbeat bill",
        "date": pd.to_datetime("2019-04-04"),
        "relevant_dates": {},
        "regex": get_event_regex(
            ABORTION_KEYWORDS,
            [
                "georgia",
                "heartbeat bill",
                "six week",
                "constitutional",
            ],
            "and",
        ),
    },
}
