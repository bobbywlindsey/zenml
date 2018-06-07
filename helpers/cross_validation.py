from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# kfold is already implemented fo sklearn models
from sklearn.model_selection import cross_val_score as kfold


# Construct your own validation and test sets if modeling a time series problem or
# to ensure the validation and test sets come from the same distribution

def train_test_data(dataframe, target_variable_name, train_size=.8, random_state=42):
    """
    Returns dataframes of training and test sets
    Use this function if implementing cross validation with sklearn
    :param dataframe: pandas.DataFrame
    :param target_variable_name: str
    :param train_size: float from 0 to 1
    :param random_state: int
    :return: tuple of pandas.DataFrames with numpy.array as the final item (train, test, train_labels, test_labels, classes)
    """
    # create design matrix and target vector y
    design_matrix = np.array(dataframe.drop(target_variable_name, axis=1))
    y = np.array(dataframe[target_variable_name])
    test_size = 1-train_size
    train_data, test_data, train_labels, test_labels = train_test_split(
        design_matrix, y, test_size=test_size, stratify=y, random_state=random_state)
    # convert splits to pandas data frames
    columns = list(dataframe.columns)
    columns.remove(target_variable_name)
    train_data = pd.DataFrame(train_data, columns=columns)
    test_data = pd.DataFrame(test_data, columns=columns)
    train_labels = pd.DataFrame(train_labels, columns=[target_variable_name])
    test_labels = pd.DataFrame(test_labels, columns=[target_variable_name])
    # return classes
    classes = np.sort(dataframe[target_variable_name].unique())
    return train_data, test_data, train_labels, test_labels, classes


def train_val_test_data(dataframe, target_variable_name, train_size=.7, val_size=.1, random_state=42):
    """
    Returns dataframes of train, validation, and test sets
    This option is used if not doing cross validation
    :param dataframe: pandas.DataFrame
    :param target_variable_name: str
    :param train_size: float from 0 to 1
    :param val_size: float from 0 to 1
    :param random_state: int
    :return: tuple of pandas.DataFrames with numpy.array as the final item (train, val, test, train_labels, val_labels, test_labels, classes)
    """
    # create design matrix and target vector y
    design_matrix = np.array(dataframe.drop(target_variable_name, axis = 1))
    y = np.array(dataframe[target_variable_name])
    test_size = 1-(train_size + val_size)
    # split data into train and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(design_matrix, y, test_size=test_size, stratify=y, random_state=random_state)
    # split train data into train and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=val_size, stratify=train_labels, random_state=random_state)
    # convert splits to pandas dataframes
    columns = list(dataframe.columns)
    columns.remove(target_variable_name)
    train_data = pd.DataFrame(train_data, columns=columns)
    val_data = pd.DataFrame(val_data, columns=columns)
    test_data = pd.DataFrame(test_data, columns=columns)
    train_labels = pd.DataFrame(train_labels, columns=[target_variable_name])
    val_labels = pd.DataFrame(val_labels, columns=[target_variable_name])
    test_labels = pd.DataFrame(test_labels, columns=[target_variable_name])
    # return unique labels
    unique_labels = np.sort(dataframe[target_variable_name].unique())
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, unique_labels
