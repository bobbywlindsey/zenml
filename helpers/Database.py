import sqlalchemy
import pymssql
import pandas as pd
import json
import numpy as np
import io
from elasticsearch import Elasticsearch
import requests


class Database:
    def __init__(self, connection):
        self.connection = connection

        try:
            with open('db_config.json') as f:
                    db_config = json.load(f)
            self.dl_hostname_vpn = db_config['dl_hostname_vpn']
            self.dl_hostname_network = db_config['dl_hostname_network']
            self.dl_username = db_config['dl_username']
            self.dl_password = db_config['dl_password']
            self.dl_database = db_config['dl_database']
            self.ms_hostname = db_config['ms_hostname']
            self.ms_username = db_config['ms_username']
            self.ms_password = db_config['ms_password']
            self.ms_database = db_config['ms_database']
            self.teamcenter = db_config['teamcenter']
            self.teamcenter_url = db_config['teamcenter_url']
        except:
            raise Exception('Could not import db_config.json')

        self.ms_engine_string = 'mssql+pymssql://' + self.ms_username + ':'\
                            + self.ms_password + '@' + self.ms_hostname + '/' + self.ms_database

        self.dl_engine_string_network = 'postgresql+psycopg2://' + self.dl_username + ':'\
                            + self.dl_password + '@' + self.dl_hostname_network \
                            + '/' + self.dl_database

        self.dl_engine_string_vpn = 'postgresql+psycopg2://' + self.dl_username + ':' \
                            + self.dl_password + '@' + self.dl_hostname_vpn \
                            + '/' + self.dl_database

        self.es = Elasticsearch(self.teamcenter)


    def create_engine(self, db_connection_string):
        """
        :param db_connection_string: str
        :return: sqlalchemy.Engine
        """
        try:
            return sqlalchemy.create_engine(db_connection_string)
        except Exception:
            raise Exception('Engine not created - check your connection')


    def mssql_read(self, query):
        """
        :param query: str
        :return: pandas.DataFrame
        """
        return pd.read_sql_query(query, self.create_engine(self.ms_engine_string))


    def mssql_update(self, query):
        """
        :param query: str
        :return: None
        """
        ms_conn_for_update = pymssql.connect(self.ms_hostname, self.ms_username,
                                             self.ms_password, self.ms_database)
        cursor = ms_conn_for_update.cursor()
        cursor.execute(query)
        ms_conn_for_update.commit()
        ms_conn_for_update.close()
        return None


    def mssql_insert_dataframe(self, dataframe, table_name):
        """
        :param dataframe: pandas.DataFrame
        :param table_name: str
        :return: None
        """
        dataframe.to_sql(table_name, self.create_engine(self.ms_engine_string),
                        if_exists='append', index=False)
        return None


    def datalake_read(self, query):
        """
        :param query: str
        :param connection: str
        :return: pandas.DataFrame
        """
        if self.connection == 'network':
            dl_engine = self.create_engine(self.dl_engine_string_network)
        else:
            dl_engine = self.create_engine(self.dl_engine_string_vpn)
        query = sqlalchemy.text(query)
        return pd.read_sql_query(query, dl_engine)


    def datalake_update(self, query):
        """
        :param query: str
        :param connection: str
        :return: None
        """
        if self.connection == 'network':
            dl_engine = self.create_engine(self.dl_engine_string_network)
        else:
            dl_engine = self.create_engine(self.dl_engine_string_vpn)
        dl_conn = dl_engine.connect()
        query = sqlalchemy.text(query)
        dl_conn.execute(query)
        dl_conn.close()
        return None


    def datalake_execute_function(self, query):
        """
        :param query: str
        :param connection: str
        :return: None
        """
        if self.connection == 'network':
            dl_engine = self.create_engine(self.dl_engine_string_network)
        else:
            dl_engine = self.create_engine(self.dl_engine_string_vpn)
        dl_conn = dl_engine.connect()
        transaction = dl_conn.begin()
        query = sqlalchemy.text(query)
        dl_conn.execute(query)
        transaction.commit()
        dl_conn.close()
        return None


    def datalake_insert_dataframe_append(self, dataframe, schema, table_name):
        """
        :param dataframe: pandas.DataFrame
        :param schema: str
        :param table_name: str
        :param connection: str
        :return: None
        """
        if self.connection == 'network':
            dl_engine = self.create_engine(self.dl_engine_string_network)
        else:
            dl_engine = self.create_engine(self.dl_engine_string_vpn)
        dataframe.to_sql(table_name, dl_engine, schema=schema, if_exists='append', index=False)
        return None


    def datalake_insert_dataframe_replace(self, dataframe, schema, table_name):
        """
        This method assumes you have already create an empty table
        with the columns of the dataframe

        :param dataframe: pandas.DataFrame
        :param schema: str
        :param table_name: str
        :param connection: str
        :return: None
        """
        if self.connection == 'network':
            dl_engine = self.create_engine(self.dl_engine_string_network)
        else:
            dl_engine = self.create_engine(self.dl_engine_string_vpn)
        dl_conn = dl_engine.raw_connection()
        cur = dl_conn.cursor()
        output = io.StringIO()
        dataframe.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        cur.copy_from(output, schema + '.' + table_name, null='')
        dl_conn.commit()
        dl_conn.close()
        return None


    def datalake_get_running_processes(self):
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
        return self.datalake_read(query)


    def datalake_kill_process(self, process_id):
        """
        :param process_id: int
        :param connection: str
        :return: None
        """
        query = """
        select pg_cancel_backend({0});
        select pg_terminate_backend({0});
        """.format(process_id)
        return self.datalake_read(query)


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


    def tc_global_search(self, query):
        """
        Get dictionary of a Team Center global search
        :param query: str
        :return: dictionary
        """
        results = self.es.search(index='teamcenter', body={
            'query': {
                'query_string': {
                    'query': query
                }
            }
        })
        print(str(results['hits']['total']) + ' results found')
        return results


    def tc_part_search(self, query):
        """
        Get dictionary of a Team Center part search
        :param query: str
        :return: dictionary
        """
        results = self.es.search(index='teamcenter', doc_type='parts', body={
            'query': {
                'query_string': {
                    'query': query
                }
            }
        })
        print(str(results['hits']['total']) + ' results found')
        return results


    def get_tc_schema(self):
        """
        Get the Team Center schema as a json object
        :return: json object
        """
        results_bytes = requests.get(self.teamcenter_url).content
        results_string = str(results_bytes, encoding='UTF-8')
        parsed = json.loads(results_string)
        return json.dumps(parsed, indent=4, sort_keys=True)