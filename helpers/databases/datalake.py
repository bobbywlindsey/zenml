import pandas as pd
import sqlalchemy
import json
import io
from ..utils import create_engine


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
    return None


def datalake_insert_dataframe_append(dataframe, schema, table_name, connection):
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
    return None


def datalake_insert_dataframe_replace(dataframe, schema, table_name, connection):
    """
    This method assumes you have already create an empty table
    with the columns of the dataframe

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
    dl_conn = dl_engine.raw_connection()
    cur = dl_conn.cursor()
    output = io.StringIO()
    dataframe.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    cur.copy_from(output, schema + '.' + table_name, null='')
    dl_conn.commit()
    dl_conn.close()
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

try:
    with open('db_config.json') as f:
            db_config = json.load(f)
    dl_hostname_vpn = db_config['dl_hostname_vpn']
    dl_hostname_network = db_config['dl_hostname_network']
    dl_username = db_config['dl_username']
    dl_password = db_config['dl_password']
    dl_database = db_config['dl_database']
except:
    raise Exception('Could not import db_config.json')

dl_engine_string_network = 'postgresql+psycopg2://' + dl_username + ':'\
                    + dl_password + '@' + dl_hostname_network \
                    + '/' + dl_database

dl_engine_string_vpn = 'postgresql+psycopg2://' + dl_username + ':' \
                    + dl_password + '@' + dl_hostname_vpn \
                    + '/' + dl_database