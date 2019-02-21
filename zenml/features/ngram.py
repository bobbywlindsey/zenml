from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


def ngram_tf(n, min_doc_freq, max_doc_freq, variables, print_stats=False):
    """
    Returns ngram dataframe from pandas series
    :param n: int
    :param min_doc_freq: float
    :param max_doc_freq: float
    :param variables: pd.Series or list of pd.Series
    :param print_stats: boolean
    :return: pd.DataFrame
    """
    # stack the variables into one if a list is passed
    if type(variables) == list:
        variables = pd.concat(variables, ignore_index=True)

    # ngram term frequency
    count_vectorizer = CountVectorizer(stop_words='english', min_df=min_doc_freq,
                                    max_df=max_doc_freq, ngram_range=(n, n))
    count_vectorizer.fit(variables)
    ngram_dict = count_vectorizer.vocabulary_
    ngram_list_sorted = sorted(
        ngram_dict.items(), key=lambda x: x[1], reverse=True)
    freq_column_name = str(n) + 'gram_tf'
    freq_perc_column_name = str(n) + 'gram_tf_perc'
    ngram_tf_df = pd.DataFrame(ngram_list_sorted, columns=[
                            'term', freq_column_name])
    ngram_tf_df[freq_perc_column_name] = ngram_tf_df[freq_column_name].apply(
        lambda x: x/ngram_tf_df.shape[0])

    # ngram inverse document frequency
    idf_vec = TfidfVectorizer(stop_words='english', min_df=min_doc_freq,
                            max_df=max_doc_freq, ngram_range=(n, n))
    x = idf_vec.fit_transform(variables)

    if print_stats:
        print('sparse matrix shape:', x.shape)
        print('nonzero count:', x.nnz)
        sparsity = 100.0 * x.nnz / (x.shape[0] * x.shape[1])
        print('sparsity: %.2f%%' % sparsity)
    # average ngram idf for each row
    weights = np.asarray(x.mean(axis=0)).ravel().tolist()
    idf_column_name = str(n) + 'gram_idf'
    ngram_tf_idf_df = pd.DataFrame(
        {'term': idf_vec.get_feature_names(), idf_column_name: weights})

    # merge the two dataframes
    ngram_tf_idf = pd.merge(
        ngram_tf_df, ngram_tf_idf_df, on='term', how='outer')
    return ngram_tf_idf


def ngram_idf_sum(variable, ngram_tf_df, n):
    """
    Returns a series of floats derived from the average
    ngram idf values for each element in the series
    :param variable: pd.Series
    :param ngram_tf_df: pd.DataFrame
    :param n: int
    :return: pd.Series
    """
    return variable.apply(lambda x: _ngram_idf_sum_on_string(x, ngram_tf_df, n))


def _ngram_idf_sum_on_string(string, ngram_tf_df, n):
    """
    Average the ngram idf values of a given string;
    meant to be used with the ngram_idf_sum function
    :param string: str
    :param ngram_tf_df: pd.DataFrame
    :param n: int
    :return: float
    """
    count_vectorizer = CountVectorizer(
        stop_words='english', ngram_range=(n, n))
    # handle empty strings
    try:
        count_vectorizer.fit([string])
        ngram_list = list(count_vectorizer.vocabulary_.keys())
        idf_df = ngram_tf_df[ngram_tf_df['term'].isin(ngram_list)]
        ngram_idf_linear_combo = idf_df.iloc[:, -1].sum()
        return ngram_idf_linear_combo
    except:
        return 0.0

