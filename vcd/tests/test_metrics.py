import unittest

from vcd.metrics import Intervals, Match, matching_metric


class IntervalTest(unittest.TestCase):

    def test_intersect_length(self):
        a = Intervals([(2, 5), (7, 8)])
        b = Intervals([(1, 3), (4, 7)])
        c = Intervals([(-1, 0), (3.5, 12)])

        self.assertAlmostEqual(2, a.intersect_length(b))  # [(2,3), (4,5)] = 2
        self.assertAlmostEqual(2.5, a.intersect_length(c))  # [(3.5, 5), (7,8)] = 2.5


class MatchMetricTest(unittest.TestCase):

    def test_perfect(self):
        """Perfect prediction."""
        gt = [Match(4, 14, 10, 18)]
        detections = [Match(4, 14, 10, 18, score=1.0)]
        self.assertAlmostEqual(1.0, matching_metric(gt, detections)[-1])

    def test_split(self):
        """Segment split across two predictions."""
        gt = [Match(4, 14, 10, 18)]
        detections = [
            Match(4, 8, 10, 14, score=1.0),
            Match(8, 14, 14, 18, score=2.0),
        ]
        self.assertAlmostEqual(1.0, matching_metric(gt, detections)[-1])

    def test_imperfect_calibrated(self):
        """A pretty good performance, reasonably well calibrated."""
        gt = [Match(4, 14, 10, 18)]
        detections = [
            Match(4, 8, 10, 14, score=1.0),
            Match(8, 14, 16, 18, score=2.0),
            Match(0, 30, 5, 25, score=0.0),  # imprecise detection comes last
        ]
        metric = matching_metric(gt, detections)[-1]
        self.assertLess(metric, 1.0)
        self.assertGreater(metric, 0.9)

    def test_imperfect_poorly_calibrated(self):
        """A pretty good performance, poorly calibrated.

        This example is the same as above, except for the score of the
        inaccurate prediction.
        """
        gt = [Match(4, 14, 10, 18)]
        detections = [
            Match(4, 8, 10, 14, score=1.0),
            Match(8, 14, 16, 18, score=2.0),
            Match(0, 30, 5, 25, score=3.0),  # miscalibrated; imprecise detection ranked first
        ]
        metric = matching_metric(gt, detections)[-1]
        self.assertLess(metric, 0.5)


if __name__ == "__main__":
    unittest.main()
