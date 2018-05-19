from .imports_and_configs import *
from .utils import *


def change_last_layer_resnet_inception(model, num_classes):
    """
    Change last layer of a resnet or inception model
    :param model: torchvision.models.module
    :param num_classes: int
    :return: torchvision.models.module
    """
    num_features_in_last_layer = model.fc.in_features
    model.fc = torch.nn.Linear(num_features_in_last_layer, num_classes)
    return model


def change_last_layer_densenet(model, num_classes):
    """
    Change last layer of a densenet model
    :param model: torchvision.models.module
    :param num_classes: int
    :return: torchvision.models.module
    """
    num_features_in_last_layer = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features_in_last_layer, num_classes)
    return model


def resnet_densenet_inception(model_name, dataloader, dataset_sizes, num_classes, freeze_all_layers=True, 
           freeze_first_n_layrs=0, learning_rate=0.001, momentum=0.9,
           num_epochs=100, step_size=7, gamma=0.1):
    """
    Resnet, Densenet, and Inception v3 models
    :param model_name: str
    :param dataloader: dictionary
    :param dataset_sizes: dictionary of train dev set sizes
    :param num_classes: int
    :param freeze_all_layers: boolean
    :param freeze_first_n_layrs: int
    :param learning_rate: float
    :param momentum: float
    :param num_epochs: int
    :param step_size: int
    :param gamma: float
    """
    if model_name == 'resnet18': model = models.resnet18(pretrained=True)
    if model_name == 'resnet34': model = models.resnet34(pretrained=True)
    if model_name == 'resnet50': model = models.resnet50(pretrained=True)
    if model_name == 'resnet101': model = models.resnet101(pretrained=True)
    if model_name == 'resnet152': model = models.resnet152(pretrained=True)
    if model_name == 'densenet121': model = models.densenet121(pretrained=True)
    if model_name == 'densenet161': model = models.densenet161(pretrained=True)
    if model_name == 'densenet169': model = models.densenet169(pretrained=True)
    if model_name == 'densenet201': model = models.densenet201(pretrained=True)
    if model_name == 'inceptionv3':
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False
    
    if freeze_all_layers:
        model = freeze_all_layers(model)
    if freeze_first_n_layrs != 0:
        model = freeze_first_n_layers(model, freeze_first_n_layrs)
    # change last layer since we'll have different number of classes
    if 'resnet' in model_name: model = change_last_layer_resnet_inception(model, num_classes)
    if 'inceptionv3' in model_name: model = change_last_layer_resnet_inception(model, num_classes)
    if 'densenet' in model_name: model = change_last_layer_densenet(model, num_classes)

    # push model to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=momentum)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    model = train_model(model, loss, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, num_epochs=num_epochs)
    return model


def change_last_layer_vgg(model):
    """
    Change last layer of a vgg model
    :param model: torchvision.models.module
    :return: torchvision.models.module
    """
    # num filters in bottleneck layer
    num_filters = model.classifier[6].in_features
    # convert all layers to list and remove the last one
    features = list(model.classifier.children())[:-1]
    # add the last layer based on teh number of classes in the dataset
    features.extend([torch.nn.Linear(num_filters, num_classes)])
    # convert it into a container and add it to our model class
    model.classifier = torch.nn.Sequential(*features)
    return model


def vgg_alexnet(model_name, dataloader, dataset_sizes, num_classes, freeze_all_layers=True, 
           freeze_first_n_layrs=0, learning_rate=0.001, momentum=0.9,
           num_epochs=100, step_size=7, gamma=0.1):
    """
    VGG and AlexNet models
    :param model_name: str
    :param dataloader: dictionary
    :param dataset_sizes: dictionary of train dev set sizes
    :param num_classes: int
    :param freeze_all_layers: boolean
    :param freeze_first_n_layrs: int
    :param learning_rate: float
    :param momentum: float
    :param num_epochs: int
    :param step_size: int
    :param gamma: float
    """
    if model_name == 'vgg11': model = models.vgg11_bn(pretrained=True)
    if model_name == 'vgg13': model = models.vgg13_bn(pretrained=True)
    if model_name == 'vgg16': model = models.vgg16_bn(pretrained=True)
    if model_name == 'vgg19': model = models.vgg19_bn(pretrained=True)
    if model_name == 'alexnet': model = models.alexnet(pretrained=True)
    
    if freeze_all_layers:
        model = freeze_all_layers(model)
    if freeze_first_n_layrs != 0:
        model = freeze_first_n_layers(model, freeze_first_n_layrs)
    model = change_last_layer_vgg(model)
    # push model to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, 
                                            model.parameters())), 
                                            lr=learning_rate, momentum=momentum)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                       gamma=gamma)
    model = train_model(model, loss, optimizer, exp_lr_scheduler, dataloader,
                        dataset_sizes, num_epochs=num_epochs)


def train_model(model, criterion, optimizer, scheduler,
                dataloader, dataset_sizes, num_epochs=25):
    """
    Trains a torchvision model
    :param model: torchvision.models.module
    :param criterion: torch.nn loss function
    :param scheduler: torch.optim.lr_scheduler
    :param dataloader: dictionary
    :param dataset_sizes: dictionary of train dev set sizes
    :param num_epochs: int
    :return: torchvision.models.module
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                scheduler.step()
                # set model to training mode
                model.train()
            else:
                # set model to evaluate mode
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            # iterate over data in batches
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward prop
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward prop and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def transfer_learn(model_name, data_directory, freeze_params, optimizer_params,
                   learning_rate_params, num_epochs):
    """
    Transfer learning for resnets, densenets, vgg, and inception models
    :param model_name: str
    :param data_directory: str
    :param freeze_params: dictionary
    :param optimizer_params: dictionary
    :param learning_rate_params: dictionary
    :param num_epochs: int
    :return: torchvision.models.module
    """
    # unpack parameters
    step_size = optimizer_params['step_size']
    step_size = optimizer_params['gamma']
    learning_rate = learning_rate_params['learning_rate']
    momentum = learning_rate_params['momentum']
    freeze_all_layers = freeze_params['freeze_all_layers']
    freeze_first_n_layrs = freeze_params['freeze_first_n_layrs']
    # use model
    resnet_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    densenet_models = ['densenet121', 'densenet161', 'densenet169', 'densenet201']
    vgg_models = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    if model_name in resnet_models or model_name in densenet_models or model_name == 'inceptionv3':
        if model_name == 'inceptionv3':
            data_transforms = get_image_transform_rules(299)
        else:
            data_transforms = get_image_transform_rules(224)
        dataloader, dataset_sizes, num_classes = get_dataloader(data_directory, data_transforms)
        model = resnet_densenet_inception(model_name, dataloader, dataset_sizes, num_classes, freeze_all_layers=freeze_all_layers, 
                       freeze_first_n_layrs=freeze_first_n_layrs, learning_rate=learning_rate,
                       momentum=momentum, num_epochs=num_epochs,
                       step_size=step_size, gamma=gamma)
    if model_name in vgg_models or model_name == 'alexnet':
        data_transforms = get_image_transform_rules(224)
        dataloader, dataset_sizes, num_classes = get_dataloader(data_directory, data_transforms)
        model = vgg_alexnet(model_name, dataloader, dataset_sizes, num_classes, freeze_all_layers=freeze_all_layers, 
                       freeze_first_n_layrs=freeze_first_n_layrs, learning_rate=learning_rate,
                       momentum=momentum, num_epochs=num_epochs,
                       step_size=step_size, gamma=gamma)
    return model