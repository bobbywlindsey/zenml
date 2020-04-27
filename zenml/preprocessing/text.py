import numpy as np


def add_prefix(df, prefix):
    """
    Add prefix to all columns in a dataframe

    :param df: pandas.DataFrame
    :param prefix: str
    :return: pandas.DataFrame
    """

    return prefix + df


def add_suffix(df, suffix):
    """
    Add suffix to all columns in a dataframe

    :param df: pandas.DataFrame
    :param suffix: str
    :return: pandas.DataFrame
    """

    return df + suffix


def apply_map(df, dictionary):
    """
    More robust version of pd.Series.map() since values not in dictionary
    aren't converted to NaN values but are preserved.

    :param df: pandas.DataFrame of size (x, 1)
    :param dictionary: dict
    :return: pandas.DataFrame of size (x, 1)
    """

    # Squeeze to reshape (x, 1) to (x,)
    series_with_replaced_values = np.squeeze(df).apply(lambda string: dictionary[string] if string in dictionary else string)
    # Return dataframe with shape (x, 1)
    return series_with_replaced_values.to_frame()


def strip_whitespace(df):
    """
    Strip all whitespace from all columns in dataframe

    :return: pd.DataFrame
    """

    return df.apply(lambda x: x.str.strip())


def replace_string(df, str_or_regex_to_replace, new_string):
    """
    Remove a string from all columns in a dataframe

    :param df: pandas.DataFrame
    :param string_to_remove: str
    :return: pd.DataFrame
    """

    return df.apply(lambda x: x.str.replace(str_or_regex_to_replace, new_string))


def remove_string(df, str_or_regex_to_remove):
    """
    Remove a string from all columns in a dataframe

    :param df: pandas.Dataframe
    :param string_to_remove: str
    :return: pd.Dataframe
    """

    return replace_string(df, str_or_regex_to_remove, '')


def replace_nan_with_string(df, string):
    """
    Replace all NaN values in the dataframe with string

    :param df: pandas.DataFrame
    :return: pd.DataFrame
    """

    return df.fillna(string)


def like_float_to_int(df):
    """
    Takes dataframe of actual floats or a strings with a float representation
    and converts it to a dataframe of integers

    :param df: pd.DataFrame
    :return: pd.Series
    """

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
    return df.apply(lambda x: x.apply(robust_float_to_int))

