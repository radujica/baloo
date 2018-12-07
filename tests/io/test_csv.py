import os

import numpy as np

from baloo import read_csv
from ..core.test_frame import assert_dataframe_equal


class TestCSV(object):
    # TODO: maybe make a utils file with this and others
    _df1_path = os.path.dirname(__file__) + '/files/df1.csv'

    def test_read_csv(self, df1):
        actual = read_csv(self._df1_path).set_index('Unnamed: 0')
        expected = df1
        expected.index.name = 'Unnamed: 0'
        expected['b'] = expected['b'].astype(np.float64)

        assert_dataframe_equal(actual, expected)

    def test_to_csv(self, df1):
        path = os.path.dirname(__file__) + '/files/df1_test.csv'
        df1.evaluate().to_csv(path)

        try:
            actual = read_csv(path)
            expected = read_csv(self._df1_path)

            assert_dataframe_equal(actual, expected)
        finally:
            # make sure just created file is deleted
            os.remove(path)
