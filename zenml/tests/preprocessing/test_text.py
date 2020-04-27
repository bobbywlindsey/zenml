import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from zenml.preprocessing import (add_prefix, add_suffix, apply_map, strip_whitespace, replace_string,
                                   remove_string, replace_nan_with_string, like_float_to_int)


class TestText(unittest.TestCase):
    def test_add_prefix(self):
        df = pd.DataFrame({'male_names': ['Bobby', 'John']})
        result = pd.DataFrame({'male_names': ['mr_Bobby', 'mr_John']})
        assert_frame_equal(add_prefix(df, 'mr_'), result, check_dtype=True)


    def test_add_suffix(self):
        df = pd.DataFrame({'male_names': ['Bobby', 'John']})
        result = pd.DataFrame({'male_names': ['Bobby-male', 'John-male']})
        assert_frame_equal(add_suffix(df, '-male'), result, check_dtype=True)


    def test_apply_map(self):
        df = pd.DataFrame({'male_names': ['Bobby', 'John']})
        mapping = {'John': 'Bobby', 'Bobby': 'John'}
        result = pd.DataFrame({'male_names': ['John', 'Bobby']})
        assert_frame_equal(apply_map(df, mapping), result, check_dtype=True)


    def test_strip_whitespace(self):
        df = pd.DataFrame({'description': [' circus at the whitehouse ', 'politics suck ']})
        result = pd.DataFrame({'description': ['circus at the whitehouse', 'politics suck']})
        assert_frame_equal(strip_whitespace(df), result, check_dtype=True)


    def test_remove_string(self):
        df = pd.DataFrame({'description': ['circus at the whitehouse', 'the politics suck', 'atthehouse']})
        result = pd.DataFrame({'description': ['circus at  whitehouse', ' politics suck', 'athouse']})
        assert_frame_equal(remove_string(df, 'the'), result, check_dtype=True)


    def test_replace_nan_with_string(self):
        df = pd.DataFrame({'description': ['circus at the whitehouse', np.nan]})
        result = pd.DataFrame({'description': ['circus at the whitehouse', '']})
        assert_frame_equal(replace_nan_with_string(df, ''), result, check_dtype=True)


    def test_like_float_to_int(self):
        df = pd.DataFrame({'probs': ['1.0', '2.0', 2]})
        result = pd.DataFrame({'probs': [1, 2, 2]})
        assert_frame_equal(like_float_to_int(df), result, check_dtype=True)


if __name__ == '__main__':
    unittest.main()
