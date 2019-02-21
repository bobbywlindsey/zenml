from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def create_random_forest_param_grid(num_estimators, max_depths,
                                    min_samples_leaves, num_workers=1):
    """
    Returns a parameter grid for a random forest
    :param num_estimators: list
    :param max_depths: list
    :param min_samples_leaves: list
    :param num_workers: int
    :return: dictionary
    """
    criterion = ['gini', 'entropy']
    max_features = ['auto', 'sqrt', 'log2', 0.2]
    oob_scores = [False]

    param_grid = {'n_estimators': num_estimators,
                    'criterion': criterion,
                    'max_features': max_features,
                    'min_samples_leaf': min_samples_leaves,
                    'oob_score': oob_scores,
                    'max_depth': max_depths,
                    'n_jobs': [num_workers]}
    return param_grid


def random_forest(train, test, train_labels, test_labels, param_grid):
    """
    Get a tuned random forest model
    :param train: pandas.DataFrame
    :param test: pandas.DataFrame
    :param train_labels: pandas.DataFrame
    :param test_labels: pandas.DataFrame
    :param param_grid: dictionary
    :return: sklearn.ensemble.forest.RandomForestClassifier
    """
    rf = RandomForestClassifier()
    grid_rf = GridSearchCV(rf, param_grid, cv=10)
    grid_rf.fit(train, train_labels.values.ravel())

    best_val_accuracy = grid_rf.best_score_
    test_predictions = grid_rf.predict(test)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    best_parameters = grid_rf.best_params_
    print('Best validation accuracy: ' + str(best_val_accuracy))
    print('Test accuracy: ' + str(test_accuracy))
    print('Best model parameters: ' + str(best_parameters))
    return grid_rf.best_estimator_