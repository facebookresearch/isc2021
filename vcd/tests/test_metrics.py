import abc
import unittest

from vcd.metrics import Intervals, Match, match_metric_v1, match_metric_v2


class IntervalTest(unittest.TestCase):

    def test_intersect_length(self):
        a = Intervals([(2, 5), (7, 8)])
        b = Intervals([(1, 3), (4, 7)])
        c = Intervals([(-1, 0), (3.5, 12)])

        self.assertAlmostEqual(2, a.intersect_length(b))  # [(2,3), (4,5)] = 2
        self.assertAlmostEqual(2.5, a.intersect_length(c))  # [(3.5, 5), (7,8)] = 2.5


class MatchMetricTestBase:

    def match(self, gt, predictions):
        raise NotImplementedError()

    def test_perfect(self):
        """Perfect prediction."""
        gt = [Match(4, 14, 10, 18)]
        detections = [Match(4, 14, 10, 18, score=1.0)]
        self.assertAlmostEqual(1.0, self.match(gt, detections))

    def test_split(self):
        """Segment split across two predictions."""
        gt = [Match(4, 14, 10, 18)]
        detections = [
            Match(4, 8, 10, 14, score=1.0),
            Match(8, 14, 14, 18, score=2.0),
        ]
        self.assertAlmostEqual(1.0, self.match(gt, detections))

    def test_imperfect_calibrated(self):
        """A pretty good performance, reasonably well calibrated."""
        gt = [Match(4, 14, 10, 18)]
        detections = [
            Match(4, 8, 10, 14, score=1.0),
            Match(8, 14, 16, 18, score=2.0),
            Match(0, 30, 5, 25, score=0.0),  # imprecise detection comes last
        ]
        metric = self.match(gt, detections)
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
        metric = self.match(gt, detections)
        self.assertLess(metric, 0.5)

    def vcsl_fig4f(self):
        # Figure 4 (f) example from the VCSL paper.
        # In this case, we have two GT bounding boxes and two prediction bounding boxes.
        # Since there is no overlap between the GT and pred bboxes, the results
        # of our metric should be close to zero. However, with the initial implementation,
        # it is one. Yet, it becomes zero if we consider only the GT bboxs that overlap
        # with the predictions.
        gt = [Match(4, 14, 10, 18), Match(20, 28, 21, 29)]
        detections = [
            Match(4, 14, 21, 29, score=1.0),
            Match(20, 28, 10, 18, score=1.0),
        ]
        return self.match(gt, detections)


class MatchMetricV1Test(MatchMetricTestBase, unittest.TestCase):

    def match(self, gt, predictions):
        return match_metric_v1(gt, predictions)[-1]

    def test_vcsl_fig4f(self):
        # Not an important property, but note that this is a weakness of the v1 metric
        self.assertAlmostEqual(1.0, self.vcsl_fig4f())


class MatchMetricV2Test(MatchMetricTestBase, unittest.TestCase):

    def match(self, gt, predictions):
        return match_metric_v2(gt, predictions)

    def test_vcsl_fig4f(self):
        self.assertAlmostEqual(0.0, self.vcsl_fig4f())


if __name__ == "__main__":
    unittest.main()
