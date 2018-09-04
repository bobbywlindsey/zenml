import numpy as np
import pandas as pd
import json_tricks
import bcolz

class Util:
    def __init__(self):
        pass
    

    def exist_nan(self, series):
        """
        Checks if a series contains NaN values
        :param series: pd.Series
        :return: boolean
        """
        if type(series) != pd.Series:
            raise TypeError(series + ' is not of type pd.Series')
        return series.isnull().values.any()


    def array_to_series(self, array):
        """
        Converts numpy array to series
        :param array: array or np.array
        :return: pd.Series
        """
        if type(array) not in (list, np.ndarray):
            raise TypeError(array + ' is not of type list or np.array')
        return pd.Series(array, index=list(range(0, len(array))))


    def save_to_file(self, dataframe, file_name):
        """
        Saves dataframe as a CSV file.
        :param dataframe: pandas.DataFrame
        :param file_name: str
        :return: None
        """
        dataframe.to_csv(file_name, sep=',', encoding='utf-8', index=False)
        return None


    def write_json(self, json_obj, file_name):
        """
        Writes json object to a json file
        :param json_obj: json object
        :param file_name: str
        :return: None
        """
        with open(file_name, 'w') as f:
            json_tricks.dump(json_obj, f, indent=3)
        return None


    def read_json(self, filename):
        """
        Reads a json file
        :param filename: str
        :return: json object
        """
        with open(filename, 'r') as f:
            return json_tricks.load(f)


    def save_array(self, np_array, file_name):
        """
        Saves numpy array to a compressed file
        :param np_array: numpy.array
        :param file_name: str
        :return: None
        """
        c = bcolz.carray(np_array, rootdir=file_name, mode='w')
        c.flush()
        return None


    def load_array(self, file_name):
        """
        Load numpy array from compressed file
        :param file_name: str
        :return: numpy.array
        """
        return bcolz.open(file_name)[:]
