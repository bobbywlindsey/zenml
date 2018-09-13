import json_tricks


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