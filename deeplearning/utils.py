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


def freeze_all_layers(model):
    """
    Freezes all layers in the network
    :param model: torchvision.models.module
    :return: torchvision.models.module
    """
    for params in model.parameters():
        params.requires_grad = False
    return model


def freeze_first_n_layers(model, n):
    """
    Freezes first n layers in the network
    :param model: torchvision.models.module
    :param n: int
    :return: torchvision.models.module
    """
    count = 0
    for child_name, child in model.named_children():
        count += 1
        if count < n:
            for param_name, param in child.named_parameters():
                param.requires_grad = False
    return model


def get_image_transform_rules(input_size):
    """
    Transform train and dev images
    :param input_size: int
    :return: dictionary of torchvision.transforms.Compose objects
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'dev': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def get_dataloader(data_directory, data_transforms):
    """
    Get dataloader and other parameters necessary for the models
    :param data_directory: str
    :param data_transforms: dictionary of torchvision.transforms.Compose objects
    :return: dictionary, int, int
    """
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_directory, x),
                                              data_transforms[x])
                      for x in ['train', 'dev']}
    dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'dev']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'dev']}
    num_classes = len(image_datasets['train'].classes)
    return dataloader, dataset_sizes, num_classes


def imshow(grid, title=None):
    """
    Show grid as image
    :param tensor: torchvision grid
    :param title: str
    :return: None
    """
    grid = grid.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    grid = std * grid + mean
    grid = np.clip(grid, 0, 1)
    plt.imshow(grid)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    return None


def show_transformed_images(data_directory, data_transforms, num_images_to_show):
    """
    Show transformed images
    :param data_directory: str
    :param data_transforms: dictionary of torchvision.transforms.Compose objects
    :param num_images_to_show: int
    :return: None
    """
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_directory, x),
                                              data_transforms[x])
                      for x in ['train', 'dev']}
    dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=num_images_to_show,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'dev']}
    # get a batch of training data
    inputs, classes = next(iter(dataloader['train']))

    # make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    class_names = image_datasets['train'].classes
    imshow(out, title=[class_names[x] for x in classes])
    return None
