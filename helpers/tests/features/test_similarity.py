import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from helpers.features import variable_match, cosine_similarity


class TestSimilarity(unittest.TestCase):
    def test_variable_match(self):
        df = pd.DataFrame.from_dict({'model_1': ['6s', '10'], 'model_2': ['6s', '8']})
        result = pd.Series([1, 0], index=[0, 1])
        assert_series_equal(variable_match(df.model_1, df.model_2), result, check_dtype=True)

    
    def test_cosine_similarity(self):
        df = pd.DataFrame.from_dict({'text_1': ['exactly similar', 'This should be really really', 'entirely different', '', '', np.nan],
                                       'text_2': ['exactly similar', 'This should be really similar', 'same', 'blank vs text', '', np.nan]})
        result = pd.DataFrame.from_dict({'cosines': [1.0, 0.8452, 0.0, 0.0, 0.0, 0.0]})
        regex_exp = r'\w{2,}\b'
        assert_series_equal(cosine_similarity(df.text_1, df.text_2, regex_exp), result.cosines)


if __name__ == '__main__':
    unittest.main()
