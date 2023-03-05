"""
Load data constants
"""
import os

DATA_DIR = f"{os.getcwd()}/data"

COMMENT_DTYPES = {
    "author": "string",
    "body": "string",
    "body_cleaned": "string",
    "controversiality": "bool",
    "created_utc": "int64",
    "distinguished": "bool",
    "edited": "bool",
    "gilded": "bool",
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
]

USER_DTYPES = {
    "author": "string",
    "automoderator": "bool",
    "bot": "bool",
    "gender": "string",
    "angry": "bool",
    "anti": "bool",
    "astro": "bool",
    "dangerous": "bool",
    "doom": "bool",
    "military": "bool",
    "nobility": "bool",
    "trump": "bool",
}

ALL_USER_COLUMNS = USER_DTYPES.keys()

SUBREDDIT_DTYPES = {
    "subreddit": "string",
    "banned": "bool",
    "gun": "bool",
    "meta": "bool",
    "party": "string",
    "politician": "bool",
    "region": "string",
}

ALL_SUBREDDIT_COLUMNS = SUBREDDIT_DTYPES.keys()

CEN_SUBREDDITS = [
    "worldnews",
    "politics",
    "news",
]

SEED = 42
