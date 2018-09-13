import pandas as pd
import numpy as np
import re
from collections import Counter
from .embeddings import _cosine_similarity_numeric, _text_embedding


def cosine_similarity(series_1, series_2, regex_exp):
    """
    Returns a series of cosines 
    :param series_1: pd.Series of text
    :param series_2: pd.Series of text
    :param regex_exp: str
    :return: pd.Series
    """
    # make sure inputs are of object series
    if type(series_1) != pd.Series:
        raise ValueError(series_1 + ' must be a pandas series')
    elif series_1.dtype != 'O':
        raise ValueError(series_1 + ' must have data type "object"')
    if type(series_2) != pd.Series:
        raise ValueError(series_2 + ' must be a pandas series')
    elif series_2.dtype != 'O':
        raise ValueError(series_2 + ' must have data type "object"')
    # the two series must be of equal length
    if series_1.shape[0] != series_2.shape[0]:
        raise ValueError('The two pandas series must be of equal length')
    # lowercase all text
    series_1 = series_1.str.lower()
    series_2 = series_2.str.lower()
    # replace all np.nans with empty string
    series_1 = series_1.replace(np.nan, '', regex=True)
    series_2 = series_2.replace(np.nan, '', regex=True)
    # parse text
    regex = re.compile(regex_exp)
    parse = regex.findall
    # get text vector, unique words, and sums for each series
    vectors_1 = series_1.apply(lambda x: Counter(parse(x)))
    vectors_2 = series_2.apply(lambda x: Counter(parse(x)))
    unique_words_1 = vectors_1.apply(lambda x: set(x.keys()))
    unique_words_2 = vectors_2.apply(lambda x: set(x.keys()))
    sums_1 = np.array([np.sum([vector[key]**2 for key in vector.keys()]) for vector in vectors_1.values])
    sums_2 = np.array([np.sum([vector[key]**2 for key in vector.keys()]) for vector in vectors_2.values])
    # list of sets containing intersections of text
    intersections = unique_words_1.values & unique_words_2.values
    # calculate cosine similarity ratio
    numerators = np.array([sum([vectors_1[idx][word] * vectors_2[idx][word]
                                for word in intersection])
                                for idx, intersection in enumerate(intersections)])
    denominators = np.sqrt(sums_1 * sums_2)
    cosines = np.round(numerators/denominators, decimals=4)
    np.nan_to_num(cosines, copy=False)
    return pd.Series(cosines, name='cosines', index=range(0, len(cosines)))


def cosine_similarity_text_embedding(variable_1, variable_2, model):
    """
    Returns a pandas series of cosine similarities between the text embeddings of two variables
    :param variable_1: pandas.Series
    :param variable_2: pandas.Series
    :param model: gensim.models.word2vec.Word2Vec
    """
    return pd.Series([_cosine_similarity_numeric(_text_embedding(variable_1.iloc[i], model), _text_embedding(variable_2.iloc[i], model))
                    for i in range(0, variable_1.shape[0])])


def variable_match(series_1, series_2):
    """
    Returns a binary series representing the equality of
    pairwise elements in two series
    :param series_1: pd.Series
    :param series_2: pd.Series
    :return: pd.Series
    """
    return (series_1 == series_2) * 1


def jaccard_index(series_1, series_2, regex_exp):
    """
    Returns a series of cosines 
    :param series_1: pd.Series of text
    :param series_2: pd.Series of text
    :param regex_exp: str
    :return: pd.Series
    """
    # make sure inputs are of object series
    if type(series_1) != pd.Series:
        raise ValueError(series_1 + ' must be a pandas series')
    elif series_1.dtype != 'O':
        raise ValueError(series_1 + ' must have data type "object"')
    if type(series_2) != pd.Series:
        raise ValueError(series_2 + ' must be a pandas series')
    elif series_2.dtype != 'O':
        raise ValueError(series_2 + ' must have data type "object"')
    # the two series must be of equal length
    if series_1.shape[0] != series_2.shape[0]:
        raise ValueError('The two pandas series must be of equal length')
    # lowercase all text
    series_1 = series_1.str.lower()
    series_2 = series_2.str.lower()
    # replace all np.nans with empty string
    series_1 = series_1.replace(np.nan, '', regex=True)
    series_2 = series_2.replace(np.nan, '', regex=True)
    # parse text
    regex = re.compile(regex_exp)
    parse = regex.findall
    # get text vector, unique words, and sums for each series
    temp_df = pd.DataFrame({'series_1': series_1.apply(lambda x: parse(x)),
                            'series_2': series_2.apply(lambda x: parse(x))})
    jaccard_indices = temp_df.apply(lambda row: np.round(len(set(row[0]) & set(row[1])) / len(set(row[0]) | set(row[1])), decimals=4)
                                       if len(set(row[0]) | set(row[1])) != 0
                                       else 0.0, axis=1)
    jaccard_indices.name = 'jaccard_indices'
    return jaccard_indices
