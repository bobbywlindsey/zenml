import pymssql
import pandas as pd
import json
from ..utils import create_engine


def mssql_read(query):
    """
    :param query: str
    :return: pandas.DataFrame
    """
    return pd.read_sql_query(query, create_engine(ms_engine_string))


def mssql_update(query):
    """
    :param query: str
    :return: None
    """
    ms_conn_for_update = pymssql.connect(ms_hostname, ms_username,
                                            ms_password, ms_database)
    cursor = ms_conn_for_update.cursor()
    cursor.execute(query)
    ms_conn_for_update.commit()
    ms_conn_for_update.close()
    return None


def mssql_insert_dataframe(dataframe, table_name):
    """
    :param dataframe: pandas.DataFrame
    :param table_name: str
    :return: None
    """
    dataframe.to_sql(table_name, create_engine(ms_engine_string),
                    if_exists='append', index=False)
    return None

try:
    with open('db_config.json') as f:
            db_config = json.load(f)
    ms_hostname = db_config['ms_hostname']
    ms_username = db_config['ms_username']
    ms_password = db_config['ms_password']
    ms_database = db_config['ms_database']
except:
    raise Exception('Could not import db_config.json')

ms_engine_string = 'mssql+pymssql://' + ms_username + ':'\
                    + ms_password + '@' + ms_hostname + '/' + ms_database
