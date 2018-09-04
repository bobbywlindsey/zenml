import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
# import classes here
from helpers import Database
from helpers import Preprocess
from helpers import Feature
from helpers import Util

class Tests(unittest.TestCase):
    def test_add_prefix(self):
        preprocess = Preprocess()
        df = pd.DataFrame.from_dict({'male_names': ['Bobby', 'John']})
        result = pd.DataFrame.from_dict({'male_names': ['mr_Bobby', 'mr_John']}).male_names
        assert_series_equal(preprocess.add_prefix('mr_', df.male_names), result, check_dtype=True)


    def test_add_suffix(self):
        preprocess = Preprocess()
        df = pd.DataFrame.from_dict({'male_names': ['Bobby', 'John']})
        result = pd.DataFrame.from_dict({'male_names': ['Bobby-male', 'John-male']}).male_names
        assert_series_equal(preprocess.add_suffix('-male', df.male_names), result, check_dtype=True)


    def test_strip_whitespace(self):
        preprocess = Preprocess()
        df = pd.DataFrame.from_dict({'description': [' circus at the whitehouse ', 'politics suck ']})
        result = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', 'politics suck']}).description
        assert_series_equal(preprocess.strip_whitespace(df.description), result, check_dtype=True)


    def test_string_to_float(self):
        preprocess = Preprocess()
        df = pd.DataFrame.from_dict({'probs': ['0.3', '0.8', 2]})
        result = pd.DataFrame.from_dict({'probs': [0.3, 0.8, 2]}).probs
        assert_series_equal(preprocess.string_to_float(df.probs), result, check_dtype=True)


    def test_remove_string(self):
        preprocess = Preprocess()
        df = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', 'the politics suck', 'atthehouse']})
        result = pd.DataFrame.from_dict({'description': ['circus at whitehouse', 'politics suck', 'athouse']}).description
        assert_series_equal(preprocess.remove_string('the', df.description), result, check_dtype=True)


    def test_replace_string_with_nan(self):
        preprocess = Preprocess()
        df = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', '']})
        result = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', np.nan]}).description
        assert_series_equal(preprocess.replace_string_with_nan('', df.description), result, check_dtype=True)


    def test_replace_nan_with_string(self):
        preprocess = Preprocess()
        df = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', np.nan]})
        result = pd.DataFrame.from_dict({'description': ['circus at the whitehouse', '']}).description
        assert_series_equal(preprocess.replace_nan_with_string('', df.description), result, check_dtype=True)


    def test_like_float_to_int(self):
        preprocess = Preprocess()
        df = pd.DataFrame.from_dict({'probs': ['1.0', '2.0', 2]})
        result = pd.DataFrame.from_dict({'probs': [1, 2, 2]}).probs
        assert_series_equal(preprocess.like_float_to_int(df.probs), result, check_dtype=True)


    def test_create_dummy_variables(self):
        feature = Feature()
        df = pd.DataFrame.from_dict({'eye_color': ['blue', 'green', 'brown', 'blue']})
        prefix = 'eye_color'
        result = pd.DataFrame.from_dict({'eye_color_blue': [1, 0, 0, 1], 'eye_color_green': [0, 1, 0, 0]})
        assert_frame_equal(feature.create_dummy_variables(df.eye_color, prefix), result, check_dtype=True)


    def test_array_to_series(self):
        some_array = [1, 2, 3, 4]
        result = pd.Series(some_array, index=list(range(0, len(some_array))))
        util = Util()
        assert_series_equal(util.array_to_series(some_array), result, check_dtype=True)

    
    def test_parse_date(self):
        feature = Feature()
        util = Util()
        date_series = util.array_to_series(['2018-05-10', '1990-04-10'])
        result = pd.DataFrame.from_dict({'day_of_week': ['Thurs', 'Tues'], 'month': [5, 4]})
        assert_frame_equal(feature.parse_date(date_series), result, check_dtype=True)


    def test_ordinal_to_indicator(self):
        feature = Feature()
        array = np.array([0, 1, 2, 3])
        result = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]])
        assert_array_equal(feature.ordinal_to_indicator(array), result)


    def test_variable_match(self):
        feature = Feature()
        df = pd.DataFrame.from_dict({'model_1': ['6s', '10'], 'model_2': ['6s', '8']})
        result = pd.Series([1, 0], index=[0, 1])
        assert_series_equal(feature.variable_match(df.model_1, df.model_2), result, check_dtype=True)



if __name__ == '__main__':
    unittest.main()
