from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import re
import math
from collections import Counter
from sklearn.metrics import pairwise
from gensim.models import Word2Vec
import numpy as np


def pca(dataframe_without_target, variance_explained):
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


# Cosine similarity for term frequency

def word_frequency_dict(text):
    """
    Get a word frequency dictionary from a string
    :param text: str
    :return: collections.Counter
    """
    try:
        word_regex = re.compile(r'\w{2,}\b')
        words = word_regex.findall(text)
    except:
        words = ['']
    return Counter(words)


def cosine_similarity_text(text_1, text_2):
    """
    Get the cosine similarity between two strings
    :param text_1: str
    :param text_2: str
    :return: float
    """
    vec_1 = word_frequency_dict(text_1)
    vec_2 = word_frequency_dict(text_2)

    intersection = set(vec_1.keys()) & set(vec_2.keys())
    numerator = sum([vec_1[x] * vec_2[x] for x in intersection])
    if numerator == 0:
        return 0.0

    sum1 = sum([vec_1[x]**2 for x in vec_1.keys()])
    sum2 = sum([vec_2[x]**2 for x in vec_2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def cosine_similarity(variable_1, variable_2):
    """
    Get the cosine similarity between every element in two series
    :param variable_1: pandas.Series
    :param variable_2: pandas.Series
    :return: pandas.Series
    """
    return pd.Series([cosine_similarity_text(variable_1.iloc[i], variable_2.iloc[i])
                      for i in range(0, variable_1.shape[0])])


# Word and text embeddings

def word_embedding(variables, ngram, min_word_count, epochs, initial_learning_rate,
                   workers, model_type='cbow', hidden_layer_size=100):
    """
    Returns a word2vec model applied to one or multiple variables
    :param variables: pandas.Series or list of pandas.Series
    :param ngram: int
    :param min_word_count: int
    :param epochs: int
    :param initial_learning_rate: float
    :param workers: int
    :param model_type: str
    :param hidden_layer_size: int
    :return: gensim.models.word2vec.Word2Vec
    """
    if type(variables) == list:
        variables = pd.concat(variables, ignore_index=True)
    sentences = variables.apply(lambda x: x.lower().split()).values
    for epoch in epochs:
        if model_type == 'cbow':
            model = Word2Vec(min_count=min_word_count, workers=workers,
                             window=ngram, sg=0, size=hidden_layer_size,
                             alpha=initial_learning_rate)
        elif model_type == 'skipgram':
            model = Word2Vec(min_count=min_word_count, workers=workers,
                             window=ngram, sg=1, size=hidden_layer_size,
                             alpha=initial_learning_rate)
        else:
            raise Exception(
                model_type + ' is not an option for the model parameter')
        model.build_vocab(sentences)
        model.train(sentences, total_examples=len(sentences), epochs=epoch,
                    compute_loss=True)
        print('epoch: ' + str(epoch))
        print('training loss: ' + str(model.get_latest_training_loss()))
    return model


def text_embedding(string, model):
    """
    Returns the average of all words vectors in a string
    :param string: str
    :param model: gensim.models.word2vec.Word2Vec
    :return: numpy.array
    """
    words = string.lower().split()
    vectors = []
    for word in words:
        try:
            vectors.append(model.wv.word_vec(word))
        except:
            continue
    return np.mean(vectors, axis=0)


# Cosine similarity for text embeddings

def cosine_similarity_numeric(array_1, array_2):
    """
    Get the cosine similarity between two numpy arrays
    :param array_1: numpy.array
    :param array_2: numpy.array
    :return: float
    """
    array_1 = array_1.reshape(1, -1)
    array_2 = array_2.reshape(1, -1)
    try:
        return pairwise.cosine_similarity(array_1, array_2)[0][0]
    except:
        return 0.0


def cosine_similarity_text_embedding(variable_1, variable_2, model):
    """
    Returns a pandas series of cosine similarities between the text embeddings of two variables
    :param variable_1: pandas.Series
    :param variable_2: pandas.Series
    :param model: gensim.models.word2vec.Word2Vec
    """
    return pd.Series([cosine_similarity_numeric(text_embedding(variable_1.iloc[i], model), text_embedding(variable_2.iloc[i], model))
                      for i in range(0, variable_1.shape[0])])
