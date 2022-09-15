import unittest

import numpy as np

from sklearn.preprocessing import normalize
from vcd.baseline.candidates import CandidatePair
from vcd.baseline.localization import VCSLLocalizationMaxSim

from vcd.index import VideoFeature

try:
    import vcsl  # @manual  # noqa: F401

    HAS_VCSL = True
except ImportError:
    HAS_VCSL = False


class LocalizationTest(unittest.TestCase):
    D = 64

    def make_feature(self, i, feature, timestamps=None):
        if timestamps is None:
            timestamps = np.arange(feature.shape[0]) * 1.0
        return VideoFeature(video_id=i, feature=feature, timestamps=timestamps)

    def random_feature(self, size):
        feature = np.random.normal(size=(size, self.D))
        return normalize(feature)

    def make_test_case_1(self):
        a = self.random_feature(45)
        b = self.random_feature(30)
        c = self.random_feature(60)
        a[20:30, :] = c[30:40, :]
        queries = [self.make_feature(1, a)]
        refs = [self.make_feature(2, b), self.make_feature(3, c)]
        return queries, refs

    @unittest.skipIf(not HAS_VCSL, "VCSL required for localization test")
    def test_localize(self):
        queries, refs = self.make_test_case_1()
        localization = VCSLLocalizationMaxSim(queries, refs, "TN")
        # No matches for this pair:
        matches = localization.localize(CandidatePair(1, 2, 1.0))
        self.assertEqual(0, len(matches))
        # This pair has a match:
        matches = localization.localize(CandidatePair(1, 3, 2.0))
        self.assertGreaterEqual(len(matches), 1)

    @unittest.skipIf(not HAS_VCSL, "VCSL required for localization test")
    def test_localize_all(self):
        queries, refs = self.make_test_case_1()
        localization = VCSLLocalizationMaxSim(queries, refs, "TN")
        matches = localization.localize_all(
            [CandidatePair(1, 2, 1.0), CandidatePair(1, 3, 2.0)]
        )
        self.assertGreaterEqual(len(matches), 1)
        for match in matches:
            self.assertEqual(1, match.query_id)
            self.assertEqual(3, match.ref_id)


if __name__ == "__main__":
    unittest.main()
