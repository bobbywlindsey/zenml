"""
Do stuff to categorical features
"""

import numpy as np


def count_encode(df):
    """
    Count encode a categorical variable which replaces the categorical
    value with the number of times it occurs in the
    data set

    :param df: pandas.DataFrame of size (x, 1)
    :return: pandas.DataFrame of size (x, 1)
    """

    # Squeeze to reshape (x, 1) to (x,) and get value counts
    value_counts_series = np.squeeze(df).value_counts()
    count_mapping = value_counts_series.to_dict()
    # Squeeze to reshape (x, 1) to (x,) and apply mapping
    count_encoded_series = np.squeeze(df).map(lambda string: count_mapping[string] if string in count_mapping else string)
    # Return dataframe with shape (x, 1)
    return count_encoded_series.to_frame()
