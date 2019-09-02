import pandas as pd
import numpy as np
import IPython.display as ipd
import missingno as msno
import operator
from ..utils import (get_categorical_variable_names,
                    get_numerical_variables,
                    display)


def show_missing_data(df):
    """ returns table of completeness of each row of the data from most incomplete to most complete"""
    # diplay table for missing percentages
    num_rows = df.shape[0]
    percent_missing = {}
    for column_name in df.columns.values:
        num_missing = df[column_name].isnull().sum()
        try:
            num_missing += (df[column_name] == '').sum()
        except:
            continue
        percent_missing[column_name] = (num_missing / num_rows) * 100
    percent_missing_df = pd.DataFrame({'% missing': pd.Series(percent_missing)})
    if percent_missing_df.empty:
        print('No missing data!')
    else:
        display(percent_missing_df)
    return None


def get_categories_to_rows_ratio(df):
    """
    Gets ratio of unique categories to number of rows
    in the categorical variable; do this for each categorical
    variable 
    :param df: pd.DataFrame
    :return: array of tuples
    """
    cat_columns = get_categorical_variable_names(df)
    ratios = {col:len(df[col].unique()) / df[col].count() for col in cat_columns}
    sorted_ratios = sorted(ratios.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_ratios


def get_labels_percentage(df, target_variable_name):
    """
    Get the percentage of each label in the target variable
    as a percentage
    :param df: pd.DataFrame
    :param target_variable_name: str
    :return: pandas.Series
    """
    return df[target_variable_name].value_counts(normalize=True)

