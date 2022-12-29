"""
Pre-processing constants
"""
ALL_COMMENT_COLUMNS = [
    "author",
    "body",
    "body_cleaned",
    "controversiality",
    "created_utc",
    "distinguished",
    "edited",
    "gilded",
    "id",
    "language",
    "link_id",
    "parent_id",
    "retrieved_on",
    "score",
    "subreddit",
    "subreddit_id",
]

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

MIN_OCCURENCE_FOR_VOCAB = 25
