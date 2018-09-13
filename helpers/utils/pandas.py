import pandas as pd
import numpy as np


def get_numerical_variables(df):
    """
    Returns dataframe with just continuous variables
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    if type(df) != pd.DataFrame:
        raise TypeError(df + ' is not of type pd.DataFrame')
    return df._get_numeric_data()


def get_categorical_variable_names(df):
    """
    Returns list of categorical column names from dataframe
    :param df: pd.DataFrame
    :return: list
    """
    if type(df) != pd.DataFrame:
        raise TypeError(df + ' is not of type pd.DataFrame')
    columns = df.columns
    numerical_variable_names = list(get_numerical_variables(df).columns)
    return list(set(columns) - set(numerical_variable_names))


def get_categorical_variables(df):
    """
    Returns dataframe with just categorical variables
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    if type(df) != pd.DataFrame:
        raise TypeError(df + ' is not of type pd.DataFrame')
    return df[get_categorical_variable_names(df)]


def exist_nan(series):
    """
    Checks if a series contains NaN values
    :param series: pd.Series
    :return: boolean
    """
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    return series.isnull().values.any()


def array_to_series(array):
    """
    Converts numpy array to series
    :param array: array or np.array
    :return: pd.Series
    """
    if type(array) not in (list, np.ndarray):
        raise TypeError(array + ' is not of type list or np.array')
    return pd.Series(array, index=list(range(0, len(array))))


def save_to_file(dataframe, file_name):
    """
    Saves dataframe as a CSV file.
    :param dataframe: pandas.DataFrame
    :param file_name: str
    :return: None
    """
    dataframe.to_csv(file_name, sep=',', encoding='utf-8', index=False)
    return None
