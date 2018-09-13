import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from helpers.utils import exist_nan


def create_dummy_variables(series, prefix):
    """
    Returns dataframe with indicator variables for a given categorical series
    :param df: pd.DataFrame
    :param variable_name: str
    :param prefix: str
    :return: pd.DataFrame
    """
    # check types
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    if type(prefix) != str:
        raise TypeError (prefix + ' is not of type str')
    categories = series.unique()
    if exist_nan(series):
        categories = np.append(categories, 'NaN')
    categories = categories[:-1]
    df = pd.DataFrame()
    for category in categories:
        column_name = prefix + '_' + category
        df[column_name] = series.map(lambda x: 1 if x == category else 0)
    return df
