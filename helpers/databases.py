import sqlalchemy
import pymssql
import pandas as pd
import json
import numpy as np


def create_engine(db_connection_string):
    """
    :param db_connection_string: str
    :return: sqlalchemy.Engine
    """
    try:
        return sqlalchemy.create_engine(db_connection_string)
    except Exception as e:
        print('Engine not created - check your connection')


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
    print('updated MS SQL')
    return None


def mssql_insert_dataframe(dataframe, table_name):
    """
    :param dataframe: pandas.DataFrame
    :param table_name: str
    :return: None
    """
    dataframe.to_sql(table_name, create_engine(ms_engine_string),
                    if_exists='append', index=False)
    print("inserted rows into MS SQL's {0}".format(table_name))
    return None


def datalake_read(query, connection):
    """
    :param query: str
    :param connection: str
    :return: pandas.DataFrame
    """
    if connection == 'network':
        dl_engine = create_engine(dl_engine_string_network)
    else:
        dl_engine = create_engine(dl_engine_string_vpn)
    query = sqlalchemy.text(query)
    return pd.read_sql_query(query, dl_engine)


def datalake_update(query, connection):
    """
    :param query: str
    :param connection: str
    :return: None
    """
    if connection == 'network':
        dl_engine = create_engine(dl_engine_string_network)
    else:
        dl_engine = create_engine(dl_engine_string_vpn)
    dl_conn = dl_engine.connect()
    query = sqlalchemy.text(query)
    dl_conn.execute(query)
    dl_conn.close()
    print('updated Datalake')
    return None


def datalake_execute_function(query, connection):
    """
    :param query: str
    :param connection: str
    :return: None
    """
    if connection == 'network':
        dl_engine = create_engine(dl_engine_string_network)
    else:
        dl_engine = create_engine(dl_engine_string_vpn)
    dl_conn = dl_engine.connect()
    transaction = dl_conn.begin()
    query = sqlalchemy.text(query)
    dl_conn.execute(query)
    transaction.commit()
    dl_conn.close()
    print('executed datalake function')
    return None


def datalake_insert_dataframe(dataframe, schema, table_name, connection):
    """
    :param dataframe: pandas.DataFrame
    :param schema: str
    :param table_name: str
    :param connection: str
    :return: None
    """
    if connection == 'network':
        dl_engine = create_engine(dl_engine_string_network)
    else:
        dl_engine = create_engine(dl_engine_string_vpn)
    dataframe.to_sql(table_name, dl_engine, schema=schema, if_exists='append', index=False)
    print("inserted rows into Datalake's {0}.{1}".format(schema, table_name))
    return None


def datalake_get_running_processes(connection):
    """
    :param connection: str
    :return: pandas.DataFrame
    """
    query = """
        SELECT
          procpid AS PID,
          current_query,
          query_start,
          application_name,
          waiting,
          client_addr
        FROM
          pg_stat_activity
        where current_query <> '<IDLE>'
        ORDER BY 3;
    """
    return datalake_read(query, connection)


def datalake_kill_process(process_id, connection):
    """
    :param process_id: int
    :param connection: str
    :return: None
    """
    query = """
    select pg_cancel_backend({0});
    select pg_terminate_backend({0});
    """.format(process_id)
    return datalake_read(query, connection)


def create_update_table_script(schema_name, table_name, create_table_select_source, distributed_by_column,
                               primary_key_columns, connection):
    """
    :param schema_name: str
    :param table_name: str
    :param create_table_select_source: str
    :param distributed_by_column: str
    :param primary_key_columns: list
    :param connection: str
    :return: None

    Note that
        * table_name must already exist in the datalake
        * you should not include semi-colon in create_table_select_source
    """

    function_preamble = 'CREATE OR REPLACE FUNCTION ' + schema_name + '.update_' + table_name\
                        + '()\nRETURNS TEXT AS\n$BODY$\nBEGIN\n'
    temp_table_name = schema_name + '.' + table_name + '_temp '
    drop_temp_query = 'DROP TABLE IF EXISTS ' + temp_table_name + ';\n'
    create_temp_query = 'CREATE TABLE ' + temp_table_name + ' AS\n( ' + create_table_select_source \
                        + ' )\nDISTRIBUTED BY (' + distributed_by_column + ');\n'

    column_names = datalake_read('select * from ' + schema_name + '.' + table_name + ' limit 1', connection).columns
    setting_columns = 'update ' + schema_name + '.' + table_name + ' old\nset\n' \
                      + ',\n'.join(
        list(map(lambda x: x + ' = new.' + x, column_names))) + '\nfrom ' + temp_table_name + ' new\n'

    setting_columns_condition = ' AND '.join(list(map(lambda x: 'new.' + x + ' = old.' + x, primary_key_columns)))
    setting_columns_where_clause = 'where ' + setting_columns_condition + ';'
    null_condition = ' and '.join(list(map(lambda x: x + ' is null', primary_key_columns)))
    insert_new_rows = 'insert into ' + schema_name + '.' + table_name + '\nselect new.* from '\
                      + temp_table_name + ' new\nleft outer join ' \
                      + schema_name + '.' + table_name + ' old\non ' + setting_columns_condition \
                      + '\nwhere ' + null_condition + ';\n'
    drop_temp_table = 'drop table ' + temp_table_name + ';\n'
    function_footer = "RETURN('Success');\nEND\n$BODY$\nLANGUAGE plpgsql VOLATILE;\nALTER FUNCTION " \
                      + schema_name + ".update_" + table_name + "()" + "\nOWNER TO analytics_user;"

    script_source = function_preamble + drop_temp_query + create_temp_query + setting_columns \
                    + setting_columns_where_clause + insert_new_rows + drop_temp_table + function_footer
    print(script_source)
    return None


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


# import database config

try:
    with open('db_config.json') as f:
            db_config = json.load(f)
    dl_hostname_vpn = db_config['dl_hostname_vpn']
    dl_hostname_network = db_config['dl_hostname_network']
    dl_username = db_config['dl_username']
    dl_password = db_config['dl_password']
    dl_database = db_config['dl_database']
    ms_hostname = db_config['ms_hostname']
    ms_username = db_config['ms_username']
    ms_password = db_config['ms_password']
    ms_database = db_config['ms_database']
except:
    print('Could not import db_config.json')
    dl_hostname_vpn = ''
    dl_hostname_network = ''
    dl_username = ''
    dl_password = ''
    dl_database = ''
    ms_hostname = ''
    ms_username = ''
    ms_password = ''
    ms_database = ''

ms_engine_string = 'mssql+pymssql://' + ms_username + ':'\
                    + ms_password + '@' + ms_hostname + '/' + ms_database

dl_engine_string_network = 'postgresql+psycopg2://' + dl_username + ':'\
                    + dl_password + '@' + dl_hostname_network \
                     + '/' + dl_database

dl_engine_string_vpn = 'postgresql+psycopg2://' + dl_username + ':' \
                    + dl_password + '@' + dl_hostname_vpn \
                     + '/' + dl_database
