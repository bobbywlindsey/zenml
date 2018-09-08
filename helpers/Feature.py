import numpy as np
import pandas as pd
import re
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
from gensim.models import Word2Vec
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


    def cosine_similarity(self, series_1, series_2, regex_exp):
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
        find_all_words = regex.findall
        # get text vector, unique words, and sums for each series
        vectors_1 = series_1.apply(lambda x: Counter(find_all_words(x)))
        vectors_2 = series_2.apply(lambda x: Counter(find_all_words(x)))
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

    # Word and text embeddings

    def word_embedding(self, variables, ngram, min_word_count, epochs, initial_learning_rate,
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


    def text_embedding(self, string, model):
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


    def cosine_similarity_numeric(self, array_1, array_2):
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


    def cosine_similarity_text_embedding(self, variable_1, variable_2, model):
        """
        Returns a pandas series of cosine similarities between the text embeddings of two variables
        :param variable_1: pandas.Series
        :param variable_2: pandas.Series
        :param model: gensim.models.word2vec.Word2Vec
        """
        return pd.Series([self.cosine_similarity_numeric(self.text_embedding(variable_1.iloc[i], model), self.text_embedding(variable_2.iloc[i], model))
                        for i in range(0, variable_1.shape[0])])