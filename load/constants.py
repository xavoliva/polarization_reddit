"""
Load data constants
"""
import os

DATA_DIR = f"{os.getcwd()}/data"

COMMENT_DTYPES_PYARROW = {
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
    "body",
    "body_cleaned",
    "created_utc",
    "subreddit",
    "id",
    "parent_id",
]

USER_DTYPES_PYARROW = {
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

USER_COLUMNS = [
    "author",
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

SUBREDDIT_DTYPES_PYARROW = {
    "subreddit": "string[pyarrow]",
    "banned": "bool[pyarrow]",
    "gun": "bool[pyarrow]",
    "meta": "bool[pyarrow]",
    "party": "string[pyarrow]",
    "politician": "bool[pyarrow]",
    "region": "string[pyarrow]",
}

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

SEED = 42
