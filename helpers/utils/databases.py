import sqlalchemy


def create_engine(db_connection_string):
    """
    :param db_connection_string: str
    :return: sqlalchemy.Engine
    """
    try:
        return sqlalchemy.create_engine(db_connection_string)
    except Exception:
        raise Exception('Engine not created - check your connection')


def list_to_sql_list(self, array):
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