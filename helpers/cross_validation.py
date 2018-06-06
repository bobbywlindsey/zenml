from sklearn.model_selection import train_test_split


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
