import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import pairwise


# Word and text embeddings

def _text_embedding(string, model):
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


def _cosine_similarity_numeric(array_1, array_2):
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


def word_embedding(variables, ngram, min_word_count, epochs, initial_learning_rate,
                workers, model_type='cbow', hidden_layer_size=100):
    """
    Returns a word2vec model applied to one or more variables
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
