"""
Do stuff to categorical features
"""

import numpy as np
import pandas as pd


def count_encode(df):
    """
    Count encode categorical variable(s) which replaces the categorical
    value with the number of times it occurs in the
    data set

    :param df: numpy.ndarray
    :return: numpy.ndarray
    """

    if type(df) == pd.DataFrame:
        df = df.to_numpy()
    num_cols = df.shape[1]
    count_encoded_features = []
    for column_index in range(num_cols):
        unique_and_value_counts = np.unique(df[:, column_index], return_counts=True)
        count_mapping = dict(list(zip(unique_and_value_counts[0], unique_and_value_counts[1])))
        count_mapping_lambda = lambda string: count_mapping[string] \
             if string in count_mapping else string
        count_encoder = np.vectorize(count_mapping_lambda)
        count_encoded_feature = count_encoder(df[:, column_index])
        count_encoded_features.append(count_encoded_feature)
    return np.vstack(tuple(count_encoded_features)).T
