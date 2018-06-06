import pandas as pd
import numpy as np
import math
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Pipeline friendly functions

def add_prefix(prefix):
    """
    Returns a function that adds a prefix to a string
    :param prefix: str
    :return: function
    """
    return lambda x: prefix + str(x)


def add_suffix(suffix):
    """
    Returns a function that adds a suffix to a string
    :param suffix: str
    :return: function
    """
    return lambda x: str(x) + suffix


strip_whitespace = lambda x: x.strip() if type(x) == str else x


def replace_string_with_nan(string_to_replace):
    """
    Returns a function that replaces a string with np.nan
    :param string_to_replace: str
    :return: function
    """
    return lambda x: np.nan if str(x) == string_to_replace else x


def replace_nan_with_string(string_to_replace_nan):
    """
    Returns a function that replaces a np.nan with string
    :param string_to_replace_nan: str
    :return: function
    """
    return lambda x: string_to_replace_nan if math.isnan(x) else x


# Functions returning a dataframe

def get_numerical_variables(df):
    """
    Gets dataframe with just continuous variables
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    return df._get_numeric_data()


def get_categorical_variable_names(df):
    """
    Gets categorical column names from dataframe
    :param df: pd.DataFrame
    :return: list
    """
    columns = df.columns
    numerical_variable_names = list(get_numerical_variables(df).columns)
    return list(set(columns) - set(numerical_variable_names))


def get_categorical_variables(df):
    """
    Gets dataframe with just categorical variables
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    return df[get_categorical_variable_names(df)]


def create_dummy_variables(df, variable_name, prefix):
    """
    Gets dataframe with indicator variables for a given categorical series
    :param df: pandas.DataFrame
    :param variable_name: str
    :param prefix: str
    :return: pandas.DataFrame
    """
    categories = df[variable_name].unique()
    if exist_nan(df[variable_name]):
        categories = np.append(categories, 'NaN')
    categories = categories[:-1]
    for category in categories:
        column_name = prefix + '_' + category
        df[column_name] = df[variable_name].map(lambda x: 1 if x == category else 0)
    df = df.drop([variable_name], axis=1)
    return df


def reverse_dummy_variables(df_dummies, new_column_name):
    """
    Merge dummy variables into one column
    :param df_dummies: pandas.DataFrame
    :param new_column_name: str
    :return: pandas.DataFrame
    """
    positions = defaultdict(list)
    values = defaultdict(list)
    for i, c in enumerate(df_dummies.columns):
        if "_" in c:
            column_name, value = c.split("_", 1)
            column_name = new_column_name
            positions[column_name].append(i)
            values[column_name].append(value)
        else:
            positions["_"].append(i)
    df = pd.DataFrame({k: pd.Categorical.from_codes(
                      np.argmax(df_dummies.iloc[:, positions[k]].values, axis=1),
                      values[k])
                      for k in values})
    df[df_dummies.columns[positions["_"]]] = df_dummies.iloc[:, positions["_"]]
    return df


def parse_date(df, date_column_name):
    """
    Separates date object into separate days and months columns
    and directly applies to dataframe
    :param df: pandas.DataFrame
    :param date_column_name: str
    :return: pandas.DataFrame
    """
    date = pd.to_datetime(df[date_column_name])
    df.drop(date_column_name, axis=1, inplace=True)
    days = {0: 'Mon', 1: 'Tues', 2: 'Weds', 3: 'Thurs', 4: 'Fri', 5: 'Sat',
            6: 'Sun'}
    df['day_of_week'] = date.apply(lambda x: days[x.dayofweek])
    df['month'] = date.dt.month
    return df


# Series and numpy array functions

def ordinal_to_indicator(numpy_array):
    """
    Convert ordinal array to array of indicator arrays
    :param numpy_array: numpy.array
    :return: numpy.array
    """
    return np.array(OneHotEncoder().fit_transform(numpy_array.reshape(-1, 1)).todense())


def nominal_to_ordinal(all_labels_series, train_or_test_labels_series):
    """
    Convert nominal series or numpy array into an ordinal series or numpy array
    :param all_labels_series: pandas.Series or numpy.array
    :param train_or_test_labels_series: pandas.Series or numpy.array
    :return: pandas.Series
    """
    label_encoder = LabelEncoder()
    labels = all_labels_series.unique()
    label_encoder.fit(labels)
    return label_encoder.transform(train_or_test_labels_series)
