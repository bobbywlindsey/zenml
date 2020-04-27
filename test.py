from zenml.preprocessing import *
from zenml.utils import pipelinize

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.preprocessing import OneHotEncoder

# maybe remove
import pandas as pd


# just for data
from sklearn.datasets import load_boston
boston = load_boston()
boston_pd = pd.DataFrame(boston.data)
boston_pd.columns = boston.feature_names
boston_pd['test text'] = np.repeat('value', boston_pd.shape[0])
boston_pd['test text 2'] = np.repeat('other value', boston_pd.shape[0])
boston_pd['test text 3'] = np.repeat('bobby value', boston_pd.shape[0])
boston_pd['test text 4'] = np.repeat(' whitespace value ', boston_pd.shape[0])
boston_pd['test text 5'] = np.repeat(np.nan, boston_pd.shape[0])
boston_pd['test text 6'] = np.array(['class1', 'class2']*253)


def get_continuous_features(df):
    return list(df.select_dtypes(include=[np.number]).columns.values)


# Output of one function is input to the next
test_text_4_pipeline = Pipeline([
    ('replace', pipelinize(replace_string, 'value', 'titties')),
    ('removal', pipelinize(remove_string, 'whitespace')),
])


# Each transformation results in a new column
preprocess = ColumnTransformer([
    ('testtest4', test_text_4_pipeline, ['test text 4']),
    ('onehot', OneHotEncoder(), ['test text 6'])
    # ('prefix', pipelinize(add_prefix, 'yoyo'), ['test text', 'test text 2']),
    # # ('suffix', pipelinize(add_suffix, 'testssss'), ['test text 2']),
    # ('mapping', pipelinize(apply_map, my_mapping), ['test text 3']),
    # ('replacenan', pipelinize(replace_nan_with_string, 'NA'), ['test text 5']),
    # ('likefloat', pipelinize(like_float_to_int), ['test text 4']),
])


print(boston_pd.head())
print(boston_pd.shape)
print(preprocess.fit_transform(boston_pd))

