import jaydebeapi
import pandas as pd
from termcolor import colored
from ..utils import get_connection_params


def create_hive_connection():
    """
    Create a connection for Hive

    :param username: str
    :param password: str
    :return: jaydebeapi.connect or None
    """

    try:
        conn = jaydebeapi.connect('org.apache.hive.jdbc.HiveDriver',
                                 hive_jdbc_url,
                                 [hive_username, hive_password],
                                 hive_jar_path,
                                 '')
        return conn
    except Exception as e:
        raise Exception(e)


def execute_query(connection, query, operation='read'):
    """
    Execute query

    :param connection: jaydebeapi.connect
    :return: pandas.Dataframe or None
    """
    cursor = connection.cursor()

    # Try to execute the query
    try:
        cursor.execute(query)
    except Exception as e:
        cursor.close()
        raise Exception(e)

    # Read the results of the query in to a dataframe
    if operation == 'read':
        data = cursor.fetchall()
        # Get column names of the data
        column_names = [each[0].split('.')[1] for each in cursor.description]
        cursor.close()
        # Convert the array of tuples to a dataframe
        df = pd.DataFrame(data, columns=column_names)
        return df
    # Or just report the execution succeeded
    elif operation == 'update' or operation == 'destroy':
        cursor.close()
        print(colored('Query executed', 'green'))
        return None
    else:
        cursor.close()
        raise Exception(colored(f'{operation} is not a valid operation', 'red'))


def hive_read(query):
    """
    Execute a select statement in Hive

    :param query: str
    :return: pandas.Dataframe
    """

    conn = create_hive_connection()

    try:
        df = execute_query(conn, query)
        return df
    except Exception as e:
        raise Exception(e)


def hive_insert(query):
    """
    Insert new data into Hive

    :param query: str
    :return: None
    """
    conn = create_hive_connection()

    execute_query(conn, query, operation='update')
    return None


connection_params = get_connection_params()
hive_jdbc_url = connection_params['hive_jdbc_url']
hive_username = connection_params['hive_username']
hive_password = connection_params['hive_password']
hive_jar_path = connection_params['hive_jar_path']
