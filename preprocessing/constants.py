"""
Pre-processing constants
"""
from typing import List

import pandas as pd


from load.constants import DATA_DIR

EVENTS_DIR = f"{DATA_DIR}/events"
OUTPUT_DIR = f"{DATA_DIR}/output"
METADATA_DIR = f"{DATA_DIR}/metadata"

MIN_OCCURENCE_FOR_VOCAB = 50