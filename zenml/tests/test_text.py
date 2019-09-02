from zenml.text import Text
import pandas as pd
import unittest
import numpy as np
from pandas.testing import assert_series_equal
from numpy.testing import assert_almost_equal


class TestText(unittest.TestCase):
    def test_regex_parse(self):
        df = pd.DataFrame({'column': ['some test', 'other test']})
        regex_exp = r'\w{2,}\b'
        text_variable = Text(df.column)
        result = text_variable.regex_parse(regex_exp)
        correct = pd.Series([['some', 'test'], ['other', 'test']])
        assert_series_equal(correct, result, check_names=False)


    def test_word_counts(self):
        df = pd.DataFrame({'column': ['some test', 'other test']})
        regex_exp = r'\w{2,}\b'
        text_variable = Text(df.column)
        result = text_variable.word_counts(regex_exp)
        correct = pd.Series([{'some': 1, 'test': 1}, {'other': 1, 'test': 1}])
        assert_series_equal(correct, result, check_names=False)


if __name__ == '__main__':
    unittest.main()