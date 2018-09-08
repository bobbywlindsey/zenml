from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import re
import math
from collections import Counter
import numpy as np

class Model:
    def __init__(self):
        pass

    # Dimensionality reduction models

    def pca(self, dataframe_without_target, variance_explained):
        """
        Gets new dataframe that has been transformed by PCA
        :param dataframe_without_target: pandas.DataFrame
        :param variance_explained: float from 0 to 1
        :return: pandas.DataFrame
        """
        pca_model = PCA(svd_solver='full', n_components=variance_explained)
        pca_model.fit(dataframe_without_target)
        print('num components: {0}'.format(len(pca_model.components_)))
        print('feature vector: {0}'.format(pca_model.components_))
        dataframe_pca = pd.DataFrame(pca_model.transform(dataframe_without_target))
        return dataframe_pca

    # Random forest

    def create_random_forest_param_grid(self, num_estimators, max_depths,
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


    def random_forest(self, train, test, train_labels, test_labels, param_grid):
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
