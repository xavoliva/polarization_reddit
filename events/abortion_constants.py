
import pandas as pd

from events.utils import get_event_regex

# ABORTION
# https://19thnews.org/2021/12/abortion-in-america-photos-visual-timeline/

# https://en.wikipedia.org/wiki/Abortion_debate
ABORTION_KEYWORDS = [
    "abortion",
    "planned parenthood",
    "roe vs wade",
    "pro choice",
    "pro life",
    "anti choice",
    "anti life",
    "infanticide",
]

# TODO: ADD RELEVANT EVENT KEYWORDS
ABORTION_EVENTS_INFO = {
    "dr_george_tiller_shooting": {
        "name": "Dr. George Tiller shooting",
        "date": pd.to_datetime("2009-05-31"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "north_dakota_heartbeat_bill": {
        "name": "North Dakota heartbeat bill",
        "date": pd.to_datetime("2013-03-26"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "texas_house_bill_2": {
        "name": "Texas House Bill 2",
        "date": pd.to_datetime("2013-07-18"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "colorado_springs_shooting": {
        "name": "Colorado Springs shooting",
        "date": pd.to_datetime("2015-11-27"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "womans_health_vs_hellerstedt": {
        "name": "Whole Woman's Health v. Hellerstedt",
        "date": pd.to_datetime("2016-06-27"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "ohio_heartbeat_bill": {
        "name": "Ohio heartbeat bill",
        "date": pd.to_datetime("2019-11-04"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
    "georgia_heartbeat_bill": {
        "name": "Georgia heartbeat bill",
        "date": pd.to_datetime("2019-04-04"),
        "relevant_dates": {},
        "regex": get_event_regex(ABORTION_KEYWORDS, [], "or"),
    },
}