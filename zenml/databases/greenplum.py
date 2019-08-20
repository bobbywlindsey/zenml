import pandas as pd
import sqlalchemy
import json
import io
from ..utils import create_engine
from ..utils import get_connection_params


def greenplum_read(query):
    """
    :param query: str
    :return: pandas.DataFrame
    """
    greenplum_engine = create_engine(greenplum_engine_string)
    query = sqlalchemy.text(query)
    return pd.read_sql_query(query, greenplum_engine)


def greenplum_update(query):
    """
    :param query: str
    :return: None
    """
    greenplum_engine = create_engine(greenplum_engine_string)
    dl_conn = greenplum_engine.connect()
    query = sqlalchemy.text(query)
    dl_conn.execute(query)
    dl_conn.close()
    return None


def greenplum_execute_function(query):
    """
    :param query: str
    :return: None
    """
    greenplum_engine = create_engine(greenplum_engine_string)
    dl_conn = greenplum_engine.connect()
    transaction = dl_conn.begin()
    query = sqlalchemy.text(query)
    dl_conn.execute(query)
    transaction.commit()
    dl_conn.close()
    return None


def greenplum_insert_dataframe_append(dataframe, schema, table_name):
    """
    :param dataframe: pandas.DataFrame
    :param schema: str
    :param table_name: str
    :return: None
    """
    greenplum_engine = create_engine(greenplum_engine_string)
    dataframe.to_sql(table_name, greenplum_engine, schema=schema, if_exists='append', index=False)
    return None


def greenplum_insert_dataframe_replace(dataframe, schema, table_name):
    """
    This method assumes you have already create an empty table
    with the columns of the dataframe

    :param dataframe: pandas.DataFrame
    :param schema: str
    :param table_name: str
    :return: None
    """
    greenplum_engine = create_engine(greenplum_engine_string)
    dl_conn = greenplum_engine.raw_connection()
    cur = dl_conn.cursor()
    output = io.StringIO()
    dataframe.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    cur.copy_from(output, schema + '.' + table_name, null='')
    dl_conn.commit()
    dl_conn.close()
    return None


def greenplum_get_running_processes():
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
    return greenplum_read(query)


def greenplum_kill_process(process_id):
    """
    :param process_id: int
    :return: None
    """
    query = """
    select pg_cancel_backend({0});
    select pg_terminate_backend({0});
    """.format(process_id)
    return greenplum_read(query)


connection_params = get_connection_params()
greenplum_hostname = connection_params['greenplum_hostname']
greenplum_username = connection_params['greenplum_username']
greenplum_password = connection_params['greenplum_password']
greenplum_database = connection_params['greenplum_database']

greenplum_engine_string = 'postgresql+psycopg2://' + greenplum_username + ':'\
                    + greenplum_password + '@' + greenplum_hostname \
                    + '/' + greenplum_database
