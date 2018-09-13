import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from helpers.preprocessing import (add_prefix, add_suffix, strip_whitespace, string_to_float,
                                   remove_string, replace_string_with_nan, replace_nan_with_string,
                                   like_float_to_int)


class TestText(unittest.TestCase):
    def test_add_prefix(self):
        df = pd.DataFrame.from_dict({'male_names': ['Bobby', 'John']})
        result = pd.DataFrame.from_dict({'male_names': ['mr_Bobby', 'mr_John']}).male_names
        assert_series_equal(add_prefix('mr_', df.male_names), result, check_dtype=True)


    def test_add_suffix(self):
        df = pd.DataFrame.from_dict({'male_names': ['Bobby', 'John']})
        result = pd.DataFrame.from_dict({'male_names': ['Bobby-male', 'John-male']}).male_names
        assert_series_equal(add_suffix('-male', df.male_names), result, check_dtype=True)


    def test_strip_whitespace(self):
        df = pd.DataFrame.from_dict({'description': [' circus at the whitehouse ', 'politics suck ']})
        result = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', 'politics suck']}).description
        assert_series_equal(strip_whitespace(df.description), result, check_dtype=True)


    def test_string_to_float(self):
        df = pd.DataFrame.from_dict({'probs': ['0.3', '0.8', 2]})
        result = pd.DataFrame.from_dict({'probs': [0.3, 0.8, 2]}).probs
        assert_series_equal(string_to_float(df.probs), result, check_dtype=True)


    def test_remove_string(self):
        df = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', 'the politics suck', 'atthehouse']})
        result = pd.DataFrame.from_dict({'description': ['circus at whitehouse', 'politics suck', 'athouse']}).description
        assert_series_equal(remove_string('the', df.description), result, check_dtype=True)


    def test_replace_string_with_nan(self):
        df = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', '']})
        result = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', np.nan]}).description
        assert_series_equal(replace_string_with_nan('', df.description), result, check_dtype=True)


    def test_replace_nan_with_string(self):
        df = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', np.nan]})
        result = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', '']}).description
        assert_series_equal(replace_nan_with_string('', df.description), result, check_dtype=True)


    def test_like_float_to_int(self):
        df = pd.DataFrame.from_dict({'probs': ['1.0', '2.0', 2]})
        result = pd.DataFrame.from_dict({'probs': [1, 2, 2]}).probs
        assert_series_equal(like_float_to_int(df.probs), result, check_dtype=True)


if __name__ == '__main__':
    unittest.main()
