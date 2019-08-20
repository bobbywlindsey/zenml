import pandas as pd
import numpy as np
import sqlalchemy
import json


def create_engine(db_connection_string):
    """
    :param db_connection_string: str
    :return: sqlalchemy.Engine
    """
    try:
        return sqlalchemy.create_engine(db_connection_string)
    except Exception:
        raise Exception('Engine not created - check your connection')


def get_connection_params():
    """
    Get connection parameters for Greenplum and Hive
    :return: dict 
    """
    try:
        with open('db_config.json') as f:
            connection_params = json.load(f)
        return connection_params
    except Exception:
        raise Exception('Could not import db_config.json')


def list_to_sql_list(array):
    """
    Pass a list and get a string back ready for SQL query usage
    :param array: list, numpy.array, pandas.Series, or pandas.DataFrame
    :return: str
    """
    if isinstance(array, list) or isinstance(array, pd.Series) \
            or isinstance(array, np.ndarray):
        return str(tuple(array))
    elif isinstance(array, tuple):
        return str(array)
    else:
        raise ValueError('Input parameter datatype not supported')
