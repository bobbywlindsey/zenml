import pandas as pd
import numpy as np
import math
from collections import defaultdict


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
    return list(set(columns) - set(get_numerical_variable_names(df)))


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


#  Interrogate dataframe

def get_numerical_variable_names(df):
    """
    Gets numerical column names from dataframe
    :param df: pd.DataFrame
    :return: list
    """
    return list(get_numerical_variables(df).columns)


def exist_nan(pandas_series):
    """
    Checks if a series contains NaN values
    :param pandas_series: pandas.DataFrame
    :return: boolean
    """
    return pandas_series.isnull().values.any()


def series_contains(pandas_series, array_of_values):
    """
    Checks if a series contains a list of values
    :param pandas_series: pandas.DataFrame
    :param array_of_values: array
    :return: boolean
    """
    return not pandas_series[pandas_series.isin(array_of_values)].empty


# TODO: sort these functions
# Misc
def onehot(numpy_array):
    """
    Convert ordinal array to array of indicator arrays
    :param numpy_array: numpy.array
    :return: numpy.array
    """
    return np.array(OneHotEncoder().fit_transform(numpy_array.reshape(-1, 1)).todense())


def encode_label(all_labels_series, train_or_test_labels_series):
    """
    Encode nominal series or numpy array into an ordinal series or numpy array
    :param all_labels_series: pandas.Series or numpy.array
    :param train_or_test_labels_series: pandas.Series or numpy.array
    :return: pandas.Series
    """
    label_encoder = LabelEncoder()
    labels = all_labels_series.unique()
    label_encoder.fit(labels)
    return label_encoder.transform(train_or_test_labels_series)


def get_unique_label_to_num_rows_ratio(dataframe):
    """
    Gets ratio of unique labels to number of rows
    :param dataframe: pd.DataFrame
    :return: array of tuples
    """
    ratios = {}
    cat_columns = get_categorical_variable_names(dataframe)
    for col in cat_columns:
        ratios[col] = len(dataframe[col].unique()) / dataframe[col].count()
    ratios = sorted(ratios.items(), key=operator.itemgetter(1), reverse=True)
    return ratios


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


def pca(dataframe_without_target, variance_explained):
    """
    Gets new dataframe that has been transformed by PCA
    :param dataframe_without_target: pandas.DataFrame
    :param variance_explained: float from 0 to 1
    :return: pandas.DataFrame
    """
    pca_model = PCA(svd_solver='full', n_components=variance_explained)
    pca_model.fit(dataframe_without_target)
    print("num components: {0}".format(len(pca_model.components_)))
    print("feature vector: {0}".format(pca_model.components_))
    dataframe_pca = pd.DataFrame(pca_model.transform(dataframe_without_target))
    return dataframe_pca


# def train_test_data(dataframe, target_variable_name, train_size=.7, random_state=42):
#     """
#     Returns numpy arrays of training and test sets; this is the format sklearn uses
#     :param dataframe: pandas.DataFrame
#     :param target_variable_name: str
#     :param train_size: float from 0 to 1
#     :param random_state: int
#     :return: tuple of pandas.DataFrames with numpy.array as the final item (train, test, train_labels, test_labels, classes)
#     """
#     # create design matrix and target vector y
#     design_matrix = np.array(dataframe.drop(target_variable_name, axis = 1))
#     y = np.array(dataframe[target_variable_name])
#     test_size = 1-train_size
#     train_data, test_data, train_labels, test_labels = train_test_split(design_matrix, y, test_size=test_size, stratify=y, random_state=random_state)
#     # convert splits to pandas dataframes
#     columns = list(dataframe.columns)
#     columns.remove(target_variable_name)
#     train_data = pd.DataFrame(train_data, columns=columns)
#     test_data = pd.DataFrame(test_data, columns=columns)
#     train_labels = pd.DataFrame(train_labels, columns=[target_variable_name])
#     test_labels = pd.DataFrame(test_labels, columns=[target_variable_name])
#     # return classes
#     classes = np.sort(dataframe[target_variable_name].unique())
#     return train_data, test_data, train_labels, test_labels, classes