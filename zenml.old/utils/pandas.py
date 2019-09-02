import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# kfold is already implemented for sklearn models
from sklearn.model_selection import cross_val_score as kfold
import IPython.display as ipd


def display(design_matrix):
    """
    Pretty print for numpy arrays and series
    :param design_matrix: numpy.array or pandas.Series
    :return: None
    """
    if isinstance(design_matrix, pd.Series) or (isinstance(design_matrix, np.ndarray) and design_matrix.ndim <= 2):
        ipd.display(pd.DataFrame(design_matrix))
    else:
        ipd.display(design_matrix)
    return None


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


def get_numerical_variable_names(df):
    """
    Gets numerical column names from dataframe
    :param df: pd.DataFrame
    :return: list
    """
    return list(get_numerical_variables(df).columns)


def series_contains(pandas_series, array_of_values):
    """
    Checks if a series contains a list of values
    :param pandas_series: pandas.DataFrame
    :param array_of_values: array
    :return: boolean
    """
    return not pandas_series[pandas_series.isin(array_of_values)].empty


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
