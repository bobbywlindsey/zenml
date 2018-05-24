from .imports_and_configs import *
from .utils import *


class Net(torch.nn.Module):
    """
    Change this architecture when needed
    """

    def __init__(self, num_features, num_classes, neurons):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, neurons[0])
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(neurons[0], neurons[1])
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.out = torch.nn.Linear(neurons[1], num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


def deep_neural_net(train, train_labels, test, test_labels, neurons, num_epochs, learning_rate=0.0001, batch_size=32, save=False, problem_type='classification'):
    """
    :param train: pandas.DataFrame
    :param train_labels: pandas.DataFrame
    :param test: pandas.DataFrame
    :param test_labels: pandas.DataFrame
    :param neurons: array
    :param num_epochs: int
    :param learning_rate: float
    :param batch_size: int
    :param save: boolean
    :param problem_type: str
    :return: None
    """
    # run on GPU if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using {0} to do the number crunching'.format(device))
    # infer parameters
    num_features, num_classes, num_examples = infer_parameters(
        train, train_labels)
    # convert dataframes to torch tensors
    train, train_labels = convert_dataframe_to_tensors(train, train_labels)
    # instantiate neural net class
    neural_net = Net(num_features, num_classes, neurons)
    neural_net.to(device)
    print(neural_net)
    # create dataset to be loaded in batches
    train_data = data_utils.TensorDataset(train, train_labels)
    dataloader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    # use stochastic gradient descent and define loss function
    optimizer = torch.optim.SGD(neural_net.parameters(), lr=learning_rate)
    if problem_type == 'classification':
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = torch.nn.MSELoss()
    # batch train the neural net
    for epoch in range(num_epochs):
        for index, train_data in enumerate(dataloader, 0):
            inputs, labels = train_data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = neural_net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
        # print progress every 100 epochs
        if epoch % 100 == 0:
            print("Training cost after epoch {0}: {1:.2f}".format(
                epoch, loss.item()))
    # save model parameters
    if save:
        save_model_parameters(neural_net, 'deep_neural_net')
    print('Training finished!')
    return neural_net


def test_deep_neural_net(trained_neural_net, test, test_labels):
    """
    :param train_neural_net: torch.nn.Module
    :param test: pandas.DataFrame
    :param test_labels: pandas.DataFrame
    :return: None
    """
    # run on GPU if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_test_examples = test.shape[0]
    test, test_labels = convert_dataframe_to_tensors(test, test_labels)
    # create dataset to be loaded in batches
    test_data = data_utils.TensorDataset(test, test_labels)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=4,
                                             shuffle=False, num_workers=2)
    # calculate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_neural_net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the {0} test cases: {1:.2f}%'.format(
        num_test_examples, 100 * correct / total))
    return None
