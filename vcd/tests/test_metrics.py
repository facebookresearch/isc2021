import io
import tempfile
import unittest

import numpy as np
from vcd.metrics import (
    average_precision,
    CandidatePair,
    evaluate_matching_track,
    Intervals,
    Match,
    match_metric_v1,
    match_metric_v2,
)


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
            Match(
                0, 30, 5, 25, score=3.0
            ),  # miscalibrated; imprecise detection ranked first
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
        return match_metric_v2(gt, predictions).ap

    def test_vcsl_fig4f(self):
        self.assertAlmostEqual(0.0, self.vcsl_fig4f())

    def test_multiple_pairs(self):
        gt = [Match(4, 14, 10, 18, query_id="Q1", ref_id="R2")]
        detections = [
            Match(4, 14, 10, 18, score=3.0, query_id="Q2", ref_id="R2"),
            Match(4, 14, 10, 18, score=2.0, query_id="Q1", ref_id="R1"),
            Match(4, 14, 10, 18, score=1.0, query_id="Q1", ref_id="R2"),
        ]
        metric = self.match(gt, detections)
        # AP ~= 1/3, since precision is 1/3 by the time the lowest-score
        # correct pair prediction is seen.
        self.assertAlmostEqual(metric, 1 / 3.0)


class EvaluateMatchingTrackTest(unittest.TestCase):
    def run_test(self, gt, detections) -> float:
        with tempfile.NamedTemporaryFile() as gt_file:
            with tempfile.NamedTemporaryFile() as detection_file:
                Match.write_csv(gt, gt_file.name)
                Match.write_csv(detections, detection_file.name)
                metrics = evaluate_matching_track(gt_file.name, detection_file.name)
                return metrics.segment_ap_v2.ap

    def run_test_inline(self, gt_str, detections_str) -> float:
        with tempfile.NamedTemporaryFile("wt") as gt_file:
            with tempfile.NamedTemporaryFile("wt") as detection_file:
                gt_file.write(gt_str)
                gt_file.flush()
                detection_file.write(detections_str)
                detection_file.flush()
                metrics = evaluate_matching_track(gt_file.name, detection_file.name)
                return metrics.segment_ap_v2.ap

    def test_multiple_pairs(self):
        gt = [Match(4, 14, 10, 18, query_id=1, ref_id=2)]
        detections = [
            Match(4, 14, 10, 18, score=3.0, query_id=2, ref_id=2),
            Match(4, 14, 10, 18, score=2.0, query_id=1, ref_id=1),
            Match(4, 14, 10, 18, score=1.0, query_id=1, ref_id=2),
        ]
        metric = self.run_test(gt, detections)
        self.assertAlmostEqual(metric, 1 / 3.0)

    def test_multiple_pairs_inline(self):
        # Score column not specified (not needed for GT)
        gt = """query_start,query_end,ref_start,ref_end,query_id,ref_id
4,14,10,18,1,2
"""
        # Columns in a different order
        predictions = """query_id,ref_id,query_start,query_end,ref_start,ref_end,score
2,2,4,14,10,18,3.0
1,1,4,14,10,18,2.0
1,2,4,14,10,18,1.0
"""
        metric = self.run_test_inline(gt, predictions)
        self.assertAlmostEqual(metric, 1 / 3.0)


class EvaluateDescriptorTrackTest(unittest.TestCase):
    def ap(self, gt, predictions):
        return average_precision(gt, predictions).ap

    def test_uap(self):
        C = CandidatePair
        gt = [C(1, 10, 1.0), C(2, 11, 1.0)]
        self.assertEqual(
            1.0, self.ap(gt, [C(1, 10, 8.0), C(2, 11, 4.0), C(99, 99, 2.0)])
        )
        self.assertAlmostEqual(
            np.mean([1, 2 / 3]),
            self.ap(gt, [C(1, 10, 8.0), C(2, 11, 4.0), C(99, 99, 5.0)]),
        )
        self.assertAlmostEqual(
            np.mean([1, 0]),
            self.ap(gt, [C(1, 10, 3.0), C(2, 10, 2.0), C(99, 99, 1.0)]),
        )
        self.assertAlmostEqual(
            np.mean([1 / 2, 0]),
            self.ap(gt, [C(1, 10, 2.0), C(2, 10, 3.0), C(99, 99, 1.0)]),
        )

    def test_csv_serialization(self):
        C = CandidatePair
        candidates = [C(1, 10, 1.0), C(2, 11, 2.0)]
        with io.StringIO() as buf:
            CandidatePair.write_csv(candidates, buf)
            buf.seek(0)
            recovered = CandidatePair.read_csv(buf)
        self.assertEqual(candidates, recovered)


if __name__ == "__main__":
    unittest.main()
