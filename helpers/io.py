import json_tricks
import bcolz


def save_to_file(dataframe, file_name):
    """
    Saves dataframe as a CSV file.
    :param dataframe: pandas.DataFrame
    :param file_name: str
    :return: None
    """
    dataframe.to_csv(file_name, sep=',', encoding='utf-8', index=False)
    return None


def write_json(json_obj, file_name):
    """
    Writes json object to a json file
    :param json_obj: json object
    :param file_name: str
    :return: None
    """
    with open(file_name, 'w') as f:
        json_tricks.dump(json_obj, f, indent=3)
    return None


def read_json(filename):
    """
    Reads a json file
    :param filename: str
    :return: json object
    """
    with open(filename, 'r') as f:
        return json_tricks.load(f)


def save_array(np_array, file_name):
    """
    Saves numpy array to a compressed file
    :param np_array: numpy.array
    :param file_name: str
    :return: None
    """
    c = bcolz.carray(np_array, rootdir=file_name, mode='w')
    c.flush()
    return None


def load_array(file_name):
    """
    Load numpy array from compressed file
    :param file_name: str
    :return: numpy.array
    """
    return bcolz.open(file_name)[:]
