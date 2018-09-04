import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from .Util import Util

class Feature:
    def __init__(self):
        pass


    def create_dummy_variables(self, series, prefix):
        """
        Returns dataframe with indicator variables for a given categorical series
        :param df: pd.DataFrame
        :param variable_name: str
        :param prefix: str
        :return: pd.DataFrame
        """
        # check types
        if type(series) != pd.Series:
            raise TypeError(series + ' is not of type pd.Series')
        if type(prefix) != str:
            raise TypeError (prefix + ' is not of type str')
        categories = series.unique()
        util = Util()
        if util.exist_nan(series):
            categories = np.append(categories, 'NaN')
        categories = categories[:-1]
        df = pd.DataFrame()
        for category in categories:
            column_name = prefix + '_' + category
            df[column_name] = series.map(lambda x: 1 if x == category else 0)
        return df


    def parse_date(self, series):
        """
        Separates date object into separate days and months columns
        and returns dataframe
        :param df: pd.DataFrame
        :param date_column_name: str
        :return: pd.DataFrame
        """
        if type(series) != pd.Series:
            raise TypeError(series + ' is not of type pd.Series')
        date = pd.to_datetime(series)
        days = {0: 'Mon', 1: 'Tues', 2: 'Weds', 3: 'Thurs', 4: 'Fri', 5: 'Sat',
                6: 'Sun'}
        df = pd.DataFrame()
        df['day_of_week'] = date.apply(lambda x: days[x.dayofweek])
        df['month'] = date.dt.month
        return df


    def ordinal_to_indicator(self, numpy_array):
        """
        Convert ordinal array to array of indicator arrays
        :param numpy_array: np.array
        :return: np.array
        """
        if type(numpy_array) != np.ndarray:
            raise TypeError(numpy_array + ' is not of type np.array')
        return np.array(OneHotEncoder().fit_transform(numpy_array.reshape(-1, 1)).todense())


    def variable_match(self, series_1, series_2):
        """
        Returns a binary series representing the equality of
        pairwise elements in two series
        :param series_1: pd.Series
        :param series_2: pd.Series
        :return: pd.Series
        """
        return (series_1 == series_2) * 1


    def stem_variable(self, series):
        """
        Stem the text
        :param series: pd.Series
        :return: pd.Series
        """
        stemmer = SnowballStemmer('english')
        return series.map(lambda x: ' '.join([stemmer.stem(y) for y in x.decode('utf-8').split(' ')]))


    def ngram_tf(self, n, min_doc_freq, max_doc_freq, variables, print_stats=False):
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


    def ngram_idf_sum_on_string(self, string, ngram_tf_df, n):
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


    def ngram_idf_sum(self, variable, ngram_tf_df, n):
        """
        Returns a series of floats derived from the average
        ngram idf values for each element in the series
        :param variable: pd.Series
        :param ngram_tf_df: pd.DataFrame
        :param n: int
        :return: pd.Series
        """
        return variable.apply(lambda x: self.ngram_idf_sum_on_string(x, ngram_tf_df, n))
