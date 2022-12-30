"""
Python script to convert csv files to parquet files
"""

import sys
import pandas as pd
from preprocessing.utils import get_files_from_folder
from load.constants import COMMENT_COLUMNS

file_names = get_files_from_folder(f"data/{sys.argv[1]}")

if __name__ == "__main__":
    for file_name in file_names:
        print(f"{file_name[:-4]}")
        try:
            df = pd.read_csv(
                file_name,
                usecols=COMMENT_COLUMNS,
                dtype={
                    "subreddit": "string",
                    "author": "string",
                    "body_cleaned": "string",
                },
            )
            df["date"] = pd.to_datetime(df["created_utc"], unit="s").dt.date
            df.to_parquet(f"{file_name[:-4]}.parquet")
            del df
        except OSError as err:
            print(err)
