from functools import partial, reduce
import pandas as pd


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


def associate(_series, df_variable, value):
    """
    Associate a dataframe variable with a new value.
    This function avoids mutating the original dataframe in order to
    reduce side effects.
    :param _series: pandas.Series
    :param df_variable: str
    :param value: str
    :return: pandas.Series
    """
    from copy import deepcopy
    series = deepcopy(_series)
    series[df_variable] = value
    return series


def call(function, df_variable):
    """
    Apply a function to a dataframe variable (i.e. a pandas series)
    :param function: function
    :param df_variable: str
    :return: function
    """
    def apply_function(series):
        return associate(series, df_variable, function(series[df_variable]))
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



# Preprocessing

def add_prefix(prefix):
    """
    Adds a prefix to a specified pandas column and directly modifies data frame
    :param dataframe: pandas.DataFrame
    :param column_name: str
    :param prefix: str
    :return: None
    """
    return lambda x: prefix + str(x)


df = pd.read_csv('materialpairtrainingset.csv')
preprocess = df_pipeline(df, [call(add_prefix('poop'), 'material_1')])
print(preprocess['material_1'].head())