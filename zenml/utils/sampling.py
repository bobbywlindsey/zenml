"""
Helper functions for sampling data
"""

import random
import pandas as pd


def read_csv_random_sample(filename, sample_size=0.5):
    """
    Read a random sample into memory from a CSV

    :param filename: string
    :param sample_size: float [0, 1]
    :return: pandas.DataFrame
    """

    # If random from [0,1] interval is greater than
    # sample_size, the row will be skipped
    return pd.read_csv(
        filename,
        header=0,
        skiprows=lambda i: i > 0 and random.random() > sample_size
    )
