"""
Load data constants
"""
import os

DATA_DIR = f"{os.getcwd()}/data"

COMMENT_DTYPES = {
    "author": "string[pyarrow]",
    "body": "string[pyarrow]",
    "body_cleaned": "string[pyarrow]",
    "controversiality": "bool[pyarrow]",
    "created_utc": "int64[pyarrow]",
    "distinguished": "bool[pyarrow]",
    "edited": "bool[pyarrow]",
    "gilded": "bool[pyarrow]",
    "id": "string[pyarrow]",
    "language": "string[pyarrow]",
    "link_id": "string[pyarrow]",
    "parent_id": "string[pyarrow]",
    "retrieved_on": "int64[pyarrow]",
    "score": "int[pyarrow]",
    "subreddit": "string[pyarrow]",
    "subreddit_id": "string[pyarrow]",
}

ALL_COMMENT_COLUMNS = COMMENT_DTYPES.keys()

COMMENT_COLUMNS = [
    "author",
    "body_cleaned",
    "created_utc",
    "subreddit",
]

USER_DTYPES = {
    "author": "string[pyarrow]",
    "automoderator": "bool[pyarrow]",
    "bot": "bool[pyarrow]",
    "gender": "string[pyarrow]",
    "angry": "bool[pyarrow]",
    "anti": "bool[pyarrow]",
    "astro": "bool[pyarrow]",
    "dangerous": "bool[pyarrow]",
    "doom": "bool[pyarrow]",
    "military": "bool[pyarrow]",
    "nobility": "bool[pyarrow]",
    "trump": "bool[pyarrow]",
}

ALL_USER_COLUMNS = USER_DTYPES.keys()

SUBREDDIT_DTYPES = {
    "subreddit": "string[pyarrow]",
    "banned": "bool[pyarrow]",
    "gun": "bool[pyarrow]",
    "meta": "bool[pyarrow]",
    "party": "string[pyarrow]",
    "politician": "bool[pyarrow]",
    "region": "string[pyarrow]",
}

ALL_SUBREDDIT_COLUMNS = SUBREDDIT_DTYPES.keys()

CEN_SUBREDDITS = [
    "worldnews",
    "politics",
    "news",
]

SEED = 42
