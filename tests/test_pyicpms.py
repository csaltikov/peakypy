from pathlib import Path
import unittest

from peakypy import peakypy


class TestPyicpms(unittest.TestCase):
    def setUp(self) -> None:
        self.test_file = Path('100_uM_As3As5_rep2.TXT')

    def test_clean_icpms_data(self):
        observed = peakypy.clean_icpms_data(self.test_file)
        expected_cols = ["time", "cps"]
        self.assertEqual(expected_cols, observed.columns.tolist())

    def test_find_peaks(self):
        wlen = 90
        test_df = peakypy.clean_icpms_data(self.test_file)
        observed = peakypy.get_auc(test_df, self.test_file.name, plot=False, save_fig=False, wlen=wlen)
        observed_keys = list(observed.keys())
        expected_keys = ["arsenite", "arsenate", "sample_name"]
        self.assertEqual(set(expected_keys), set(observed_keys))
