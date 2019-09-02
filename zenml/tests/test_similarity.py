from zenml.similarity import Similarity
import pandas as pd
import unittest
import numpy as np
from pandas.testing import assert_series_equal
from numpy.testing import assert_almost_equal


class TestSimilarity(unittest.TestCase):
    def test_euclidian(self):
        df = pd.DataFrame({'var_1': [2, 3], 'var_2': [4, 5]})
        sim = Similarity(df.var_1, df.var_2)
        result = sim.euclidean()
        assert_almost_equal(2.8284271247461903, result)


    def test_correlation(self):
        df = pd.DataFrame({'var_1': [2, 3], 'var_2': [2, 3]})
        sim = Similarity(df.var_1, df.var_2)
        result = sim.correlation()
        assert_almost_equal(0.0, result)


    def test_hamming(self):
        df = pd.DataFrame({'var_1': [2, 3], 'var_2': [4, 5]})
        sim = Similarity(df.var_1, df.var_2)
        result = sim.hamming()
        assert_almost_equal(1.0, result)


    def test_variable_match_int(self):
        df = pd.DataFrame({'var_1': [2, 3], 'var_2': [2, 3]})
        sim = Similarity(df.var_1, df.var_2)
        result = sim.variable_match()
        correct = pd.Series([1, 1])
        assert_series_equal(correct, result, check_names=False)


    def test_variable_match_float(self):
        df = pd.DataFrame({'var_1': [2.23, 3.23], 'var_2': [2.999999, 3.23]})
        sim = Similarity(df.var_1, df.var_2)
        result = sim.variable_match()
        correct = pd.Series([0, 1])
        assert_series_equal(correct, result)


    def test_variable_match_text(self):
        df = pd.DataFrame({'var_1': ['test1', 'test2'], 'var_2': ['test1', 'test2']})
        sim = Similarity(df.var_1, df.var_2)
        result = sim.variable_match()
        correct = pd.Series([1, 1])
        assert_series_equal(correct, result)


    def test_variable_match_text_with_nan(self):
        df = pd.DataFrame({'var_1': [np.nan, None], 'var_2': [np.nan, None]})
        sim = Similarity(df.var_1, df.var_2)
        result = sim.variable_match()
        correct = pd.Series([0, 0])
        assert_series_equal(correct, result)


if __name__ == '__main__':
    unittest.main()
