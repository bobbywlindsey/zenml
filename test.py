from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import inspect
import numpy as np

# maybe remove
from sklearn.pipeline import Pipeline
import pandas as pd


# just for data
from sklearn.datasets import load_boston
boston = load_boston()
boston_pd = pd.DataFrame(boston.data)
boston_pd.columns = boston.feature_names
boston_pd['test text'] = np.repeat('value', boston_pd.shape[0])
boston_pd['test text 2'] = np.repeat('other value', boston_pd.shape[0])
boston_pd['test text 3'] = np.repeat('bobby value', boston_pd.shape[0])


def pipelinize(function, *args):
    keys = inspect.getfullargspec(function).args[1:]
    values = list(args)
    return FunctionTransformer(function, validate=False, kw_args=dict(zip(keys, values)))


# Series tranformations; works across rows and columns
def add_prefix(one_column_df, prefix):
    return prefix + one_column_df


def add_suffix(one_column_df, suffix):
    return one_column_df + suffix


def get_continuous_features(df):
    return list(df.select_dtypes(include=[np.number]).columns.values)


def apply_map(one_column_df, dictionary):
    """
    More robust version of pd.Series.map() since values not in dictionary
    aren't converted to NaN values but are preserved.

    :param one_column_df: pandas.DataFrame of size (x, 1)
    :param dictionary: dict
    :return: pandas.DataFrame of size (x, 1)
    """
    # Squeeze to reshape (x, 1) to (x,)
    series_with_replaced_values = np.squeeze(one_column_df).apply(lambda string: dictionary[string] if string in dictionary else string)
    # Return dataframe with shape (x, 1)
    return series_with_replaced_values.to_frame()


my_mapping = {'bobby value': 'alicia value'}

preprocess = ColumnTransformer([
    ('prefix', pipelinize(add_prefix, 'yoyo'), ['test text', 'test text 2']),
    # ('suffix', pipelinize(add_suffix, 'testssss'), ['test text 2']),
    ('mapping', pipelinize(apply_map, my_mapping), ['test text 3'])
])



print(boston_pd.head())
print(preprocess.fit_transform(boston_pd))
