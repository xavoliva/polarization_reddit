"""
Load data constants
"""

DATA_DIR = "/workspaces/polarization_reddit/data"
# DATA_DIR = "/home/xavi_oliva/Documents/Github/polarization_reddit/data"

COMMENT_DTYPES = {
    "author": "string",
    "body": "string",
    "body_cleaned": "string",
    "controversiality": "int",
    "created_utc": "int64",
    "distinguished": "int",
    "edited": "int",
    "gilded": "int",
    "id": "string",
    "language": "string",
    "link_id": "string",
    "parent_id": "string",
    "retrieved_on": "int64",
    "score": "int",
    "subreddit": "string",
    "subreddit_id": "string",
}

ALL_COMMENT_COLUMNS = COMMENT_DTYPES.keys()

COMMENT_COLUMNS = [
    "author",
    "body_cleaned",
    "created_utc",
    "subreddit",
    "subreddit_id",
]

ALL_USER_COLUMNS = [
    "author",
    "automoderator",
    "bot",
    "gender",
    "angry",
    "anti",
    "astro",
    "dangerous",
    "doom",
    "military",
    "nobility",
    "trump",
]

ALL_SUBREDDIT_COLUMNS = [
    "subreddit",
    "banned",
    "gun",
    "meta",
    "party",
    "politician",
    "region",
]

CEN_SUBREDDITS = [
    "worldnews",
    "politics",
    "news",
]
