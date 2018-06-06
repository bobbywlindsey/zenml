from functools import partial, reduce
import pandas as pd
import numpy as np


# Composition of functions on a pandas dataframe

def df_pipeline(df, functions):
    """
    Composes a list of functions on every row in a
    pandas dataframe
    :param df: pandas.DataFrame
    :param functions: list
    :return: pandas.DataFrame
    """
    rows = [df.iloc[index] for index in range(0, df.shape[0])]
    return pd.DataFrame(reduce(lambda f, g: list(map(g, f)), functions, rows))


def associate(_series, df_variables, values):
    """
    Associate dataframe variable(s) with a new value.
    This function avoids mutating the original dataframe in order to
    reduce side effects.
    :param _series: pandas.Series
    :param df_variables: str or list
    :param values: str or list
    :return: pandas.Series
    """
    from copy import deepcopy
    series = deepcopy(_series)
    # this works whether or not df_variables is str or list
    series[df_variables] = values
    return series


def call(function, df_variables):
    """
    Apply a function to a dataframe variable (i.e. a pandas series)
    :param function: function
    :param df_variables: str or list
    :return: function
    """
    if type(df_variables) not in [list, np.ndarray]:
        df_variables = [df_variables]
    def apply_function(series):
        return associate(series, df_variables, [function(series[df_variable])
                                                for df_variable in df_variables])
    return apply_function


# Composition of functions on a list

def list_pipeline(some_list, functions):
    """
    Composes a list of functions on every element in
    some_list
    :param some_list: list
    :param functions: list
    :return: list
    """
    return reduce(lambda f, g: list(map(g, f)), functions, some_list)
