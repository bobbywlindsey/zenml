import pandas as pd
import numpy as np
import math
from nltk.stem.snowball import SnowballStemmer


def add_prefix(prefix, series):
    """
    Returns a pandas series that adds a prefix to a string
    :param prefix: str
    :return: pd.Series
    """
    if type(prefix) != str:
        raise TypeError(prefix + ' is not of type str')
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    return series.apply(lambda x: prefix + str(x))


def add_suffix(suffix, series):
    """
    Returns a pandas series that adds a suffix to a string
    :param suffix: str
    :return: pd.Series
    """
    if type(suffix) != str:
        raise TypeError(suffix + ' is not of type str')
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    return series.apply(lambda x: str(x) + suffix)


def strip_whitespace(series):
    """
    Returns a pandas series that strips whitespace from both
    sides of a string
    :return: pd.Series
    """
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    return series.apply(lambda x: x.strip() if type(x) == str else x)


def string_to_float(series):
    """
    Returns a string as a float. If no string is provided,
    the original element is returned
    """
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    return series.apply(lambda x: float(x) if type(x) == str else x)


def remove_string(string_to_remove, series):
    """
    Returns a pandas series that replaces string_to_remove with ''
    :param string_to_remove: str
    :return: pd.Series
    """
    if type(string_to_remove) != str:
        raise TypeError(string_to_remove + ' is not of type str')
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    return series.apply(lambda x: ' '.join(x.replace(string_to_remove, '').split()) if type(x) == str else x)


def replace_string_with_nan(string_to_replace, series):
    """
    Returns a pandas series that replaces a string with np.nan
    :param string_to_replace: str
    :return: pd.Series
    """
    if type(string_to_replace) != str:
        raise TypeError(string_to_replace + ' is not of type str')
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    return series.apply(lambda x: np.nan if str(x) == string_to_replace else x)


def replace_nan_with_string(string_to_replace_nan, series):
    """
    Returns a pandas series that replaces a np.nan with string
    :param string_to_replace_nan: str
    :return: pd.Series
    """
    if type(string_to_replace_nan) != str:
        raise TypeError(string_to_replace_nan + ' is not of type str')
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    return series.apply(lambda x: string_to_replace_nan if ((type(x) == np.float64 or type(x) == float) and math.isnan(x)) else x)


def like_float_to_int(series):
    """
    Takes series of actual floats or a strings with a float representation
    and converts it to a series of integers
    :return: pd.Series
    """
    if type(series) != pd.Series:
        raise TypeError(series + ' is not of type pd.Series')
    def robust_float_to_int(x):
        if type(x) == str:
            try:
                return int(float(x))
            except:
                return x
        elif type(x) ==  float:
            return int(x)
        else:
            return x
    return series.apply(lambda x: robust_float_to_int(x))

def stem_variable(series):
    """
    Stem the text
    :param series: pd.Series
    :return: pd.Series
    """
    stemmer = SnowballStemmer('english')
    return series.map(lambda x: ' '.join([stemmer.stem(y) for y in x.decode('utf-8').split(' ')]))