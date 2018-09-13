import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from helpers.preprocessing import parse_date
from helpers.utils import array_to_series


class TestDates(unittest.TestCase):
    def test_parse_date(self):
        date_series = array_to_series(['2018-05-10', '1990-04-10'])
        result = pd.DataFrame.from_dict({'day_of_week': ['Thurs', 'Tues'], 'month': [5, 4]})
        assert_frame_equal(parse_date(date_series), result, check_dtype=True)

if __name__ == '__main__':
    unittest.main()
