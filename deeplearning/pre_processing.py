from .imports_and_configs import *


def infer_parameters(df_features, df_labels):
    """
    :param df_features: pandas.DataFrame
    :param df_labels: pandas.DataFrame
    :return (int, int, int)
    """
    num_features = df_features.shape[1]
    num_classes = len(np.unique(df_labels.values))
    num_examples = df_features.shape[0]
    return num_features, num_classes, num_examples


def convert_dataframe_to_tensors(df_features, df_labels):
    """
    :param df_features: pandas.DataFrame
    :param df_labels: pandas.DataFrame
    :return: (torch.autograd.Variable, torch.autograd.Variable)
    """
    df_features = torch.from_numpy(df_features.values).type(torch.FloatTensor)
    df_labels = torch.from_numpy(
        df_labels.values.reshape((df_labels.shape[0],)))
    return Variable(df_features), Variable(df_labels)


def save_model_parameters(model, filename):
    """
    :param model: torch.nn.Module
    :return: None
    """
    torch.save(model.state_dict(), './' + filename + '.pkl')
    return None


def load_model_parameters(model, pickle_file_path):
    """
    :param model: torch.nn.Module
    :param pickle_file_path: str
    :return: torch.nn.Module
    """
    return model.load_state_dict(torch.load(pickle_file_path))
