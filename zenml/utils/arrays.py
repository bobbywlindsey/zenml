import bcolz


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