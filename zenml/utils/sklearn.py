"""
Sklearn utilities
"""

import inspect
from sklearn.preprocessing import FunctionTransformer
import pandas as pd


def pipelinize(function, *args):
    """
    Take a custom function and make it compatible with sklearn's pipeline
    """

    keys = inspect.getfullargspec(function).args[1:]
    values = list(args)
    return FunctionTransformer(function, validate=False, kw_args=dict(zip(keys, values)))


def fit_column_transformer(column_transformer, data_frame, to_df=False):
    """
    Fit a sklearn ColumnTransformer with the option of returning
    and pandas data frame

    :param data_frame: sklearn.compose.ColumnTransformer
    :return: pandas.DataFrame or numpy.ndarray
    """

    df_column_names = []
    for transformer in column_transformer.transformers:
        for _ in transformer[2]:
            steps = [step[0] for step in transformer[1].steps]
            df_column_names.append(' + '.join(steps))
    fitted_column_transformer = column_transformer.fit_transform(data_frame)
    if to_df:
        return pd.DataFrame(fitted_column_transformer, columns=df_column_names)
    return fitted_column_transformer
