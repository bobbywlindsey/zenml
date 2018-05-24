from .imports_and_configs import *


# data frame manipulations
def train_test_data(dataframe, target_variable_name, train_size=.7, random_state=42):
    """
    Returns numpy arrays of training and test sets; this is the format sklearn uses
    :param dataframe: pandas.DataFrame
    :param target_variable_name: str
    :param train_size: float from 0 to 1
    :param random_state: int
    :return: tuple of pandas.DataFrames with numpy.array as the final item (train, test, train_labels, test_labels, classes)
    """
    # create design matrix and target vector y
    design_matrix = np.array(dataframe.drop(target_variable_name, axis = 1))
    y = np.array(dataframe[target_variable_name])
    test_size = 1-train_size
    train_data, test_data, train_labels, test_labels = train_test_split(design_matrix, y, test_size=test_size, stratify=y, random_state=random_state)
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


def add_prefix_to_column(dataframe, column_name, prefix):
    """
    Adds a prefix to a specified pandas column and directly modifies data frame
    :param dataframe: pandas.DataFrame
    :param column_name: str
    :param prefix: str
    :return: None
    """
    dataframe[column_name] = prefix + dataframe[column_name].astype(str)
    return None


def add_suffix_to_column(dataframe, column_name, suffix):
    """
    Adds a suffix to a specified column and directly modifies data frame
    :param dataframe: pandas.DataFrame
    :param column_name: str
    :param suffix: str
    :return: None
    """
    dataframe[column_name] = dataframe[column_name].astype(str) + suffix
    return None


def get_continuous_variables(dataframe):
    """
    Get data frame with just continuous variables
    :param dataframe: pandas.DataFrame
    :return: pandas.DataFrame
    """
    return dataframe._get_numeric_data()


def get_categorical_variables(dataframe):
    """
    Get data frame with just categorical variables
    :param dataframe: pandas.DataFrame
    :return: pandas.DataFrame
    """
    columns = dataframe.columns
    numerical_columns = dataframe._get_numeric_data().columns
    return dataframe[list(set(columns) - set(numerical_columns))]


def get_categorical_column_names(dataframe):
    """
    Get categorical column names from data frame
    :param dataframe: pd.DataFrame
    :return: list
    """
    columns = dataframe.columns
    numerical_columns = dataframe._get_numeric_data().columns
    return list(set(columns) - set(numerical_columns))


def replace_string_with_nan(dataframe, string_to_replace):
    """
    Get data frame where string of choice is replace with np.nan
    :param dataframe: pandas.DataFrame
    :param string_to_replace: str
    :return: pandas.DataFrame
    """
    return dataframe.replace({string_to_replace: np.nan}, regex=True)


def replace_nan_with_string(dataframe, string_to_replace_nan):
    """
    Get data frame where string of choice is replace with np.nan.
    :param dataframe: pandas.DataFrame
    :param string_to_replace_nan: str
    :return: panads.DataFrame
    """
    return dataframe.replace({np.nan: string_to_replace_nan}, regex=True)


def exist_nan(pandas_series):
    """
    Checks if a series contains NaN values
    :param pandas_series: pandas.DataFrame
    :return: pandas.DataFrame
    """
    return pandas_series.isnull().values.any()


def series_contains(pandas_series, array_of_values):
    """
    Check if a series contains a list of values
    :param pandas_series: pandas.DataFrame
    :param array_of_values: array
    :return: boolean
    """
    return not pandas_series[pandas_series.isin(array_of_values)].empty


def create_dummy_variables(dataframe, variable_name, prefix):
    """
    Get data frame with indicator variables for a given categorical series
    :param dataframe: pandas.DataFrame
    :param variable_name: str
    :param prefix: str
    :return: pandas.DataFrame
    """
    categories = dataframe[variable_name].unique()
    if exist_nan(dataframe[variable_name]):
        categories = np.append(categories, 'NaN')
    categories = categories[:-1]
    for category in categories:
        column_name = prefix + '_' + category
        dataframe[column_name] = dataframe[variable_name].map(lambda x: 1 if x == category else 0)
    dataframe = dataframe.drop([variable_name], axis=1)
    return dataframe


def reverse_dummy_variables(dataframe_dummies, new_column_name):
    """
    Merge dummy variables into one column
    :param dataframe_dummies: pandas.DataFrame
    :param new_column_name: str
    :return: pandas.DataFrame
    """
    positions = defaultdict(list)
    values = defaultdict(list)
    for i, c in enumerate(dataframe_dummies.columns):
        if "_" in c:
            column_name, value = c.split("_", 1)
            column_name = new_column_name
            positions[column_name].append(i)
            values[column_name].append(value)
        else:
            positions["_"].append(i)
    dataframe = pd.DataFrame({k: pd.Categorical.from_codes(
                      np.argmax(dataframe_dummies.iloc[:, positions[k]].values, axis=1),
                      values[k])
                      for k in values})
    dataframe[dataframe_dummies.columns[positions["_"]]] = dataframe_dummies.iloc[:, positions["_"]]
    return dataframe


def parse_date(dataframe, date_column_name):
    """
    Separates date object into separate days and months columns
    and directly applies to data frame
    :param dataframe: pandas.DataFrame
    :param date_column_name: str
    :return: None
    """
    date = pd.to_datetime(dataframe[date_column_name])
    dataframe.drop(date_column_name, axis=1, inplace=True)
    days = {0: 'Mon', 1: 'Tues', 2: 'Weds', 3: 'Thurs', 4: 'Fri', 5: 'Sat',
            6: 'Sun'}
    dataframe['day_of_week'] = date.apply(lambda x: days[x.dayofweek])
    dataframe['month'] = date.dt.month
    return None


# sql query helpers
def list_to_sql_list(python_array):
    """
    Pass a list and get a string back ready for SQL query usage
    :param python_array: array, numpy.array, pandas.Series, or pandas.DataFrame
    :return: str
    """
    if isinstance(python_array, list) or isinstance(python_array, pd.Series) \
            or isinstance(python_array, np.ndarray):
        return str(tuple(python_array))
    elif isinstance(python_array, tuple):
        return str(python_array)
    else:
        raise ValueError('Input parameter datatype not supported')


# file operations
def save_to_file(dataframe, file_name):
    """
    Saves data frame as a CSV file.
    :param dataframe: pandas.DataFrame
    :param file_name: str
    :return: None
    """
    dataframe.to_csv(file_name, sep=',', encoding='utf-8', index=False)
    return None


def write_json(json_obj, file_name):
    """
    Writes json object to a json file
    :param json_obj: json object
    :param file_name: str
    :return: None
    """
    with open(file_name, 'w') as f:
        json_tricks.dump(json_obj, f, indent=3)
    return None


def read_json(filename):
    """
    Reads a json file
    :param filename: str
    :return: json object
    """
    with open(filename, 'r') as f:
        return json_tricks.load(f)


def save_array(np_array, file_name):
    """
    Saves numpy array to a compressed file
    :param np_array: numpy.array
    :param file_name: str
    :return: None
    """
    c = bcolz.carray(np_array, rootdir=file_name, mode='w')
    c.flush()
    return None


def load_array(file_name):
    """
    Load numpy array from compressed file
    :param file_name: str
    :return: numpy.array
    """
    return bcolz.open(file_name)[:]


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
    Get ratio of unique labels to number of rows
    :param dataframe: pd.DataFrame
    :return: array of tuples
    """
    ratios = {}
    cat_columns = get_categorical_column_names(dataframe)
    for col in cat_columns:
        ratios[col] = len(dataframe[col].unique()) / dataframe[col].count()
    ratios = sorted(ratios.items(), key=operator.itemgetter(1), reverse=True)
    return ratios


# miscellaneous helpers
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
    Get new data frame that has been transformed by PCA
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
