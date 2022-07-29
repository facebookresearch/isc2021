import enum
from math import sqrt
from typing import NamedTuple, Collection, List, Optional, Sequence, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


class Intervals:

    # Non-overlapping, ordered by interval start.
    intervals: List[Tuple[float, float]]

    def __init__(self, intervals: Optional[List[Tuple[float, float]]] = None):
        self.intervals = intervals or []
        self._dedup()

    def add(self, interval: Tuple[float, float]):
        """Add an interval."""
        self.intervals.append(interval)
        self._dedup()

    def union(self, intervals: "Intervals") -> "Intervals":
        return Intervals(self.intervals + intervals.intervals)

    def total_length(self):
        length = 0.
        for start, end in self.intervals:
            length += end - start
        return length

    def intersect_length(self, intervals: "Intervals") -> "Intervals":
        """Compute the total_length of the intersection of two Intervals.

        This works by taking the sum of their lengths, and subtracting
        the length of their union.

        |A n B| = |A| + |B| - |A U B|
        """
        union = self.union(intervals)
        return self.total_length() + intervals.total_length() - union.total_length()

    def _dedup(self):
        if len(self.intervals) <= 1:
            return
        deduped = []
        intervals = sorted(self.intervals)
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                # Overlap case
                current_end = max(end, current_end)
            else:
                # Non-overlap case
                deduped.append((current_start, current_end))
                current_start, current_end = start, end
        deduped.append((current_start, current_end))
        self.intervals = deduped

    def __str__(self):
        return str(self.intervals)

    __repr__ = __str__


class Axis(enum.Enum):
    QUERY = enum.auto()
    REF = enum.auto()


class Match(NamedTuple):
    """A ground-truth match or predicted match.

    Omits query_id and ref_id, focusing on the single pair case for visualization.
    Doesn't lose generality. A full dataset could have all queries and refs
    concatenated and metrics would be equivalent.
    """
    query_start: float
    query_end: float
    ref_start: float
    ref_end: float
    score: float = 1.0
    query_id: str = ""
    ref_id: str = ""

    def interval(self, axis: Axis) -> Tuple[float, float]:
        if axis == Axis.QUERY:
            return (self.query_start, self.query_end)
        else:
            return (self.ref_start, self.ref_end)

    def intersection_area(self, bbox: "Match") -> float:
        # Compute the intersection boarders
        inter_q_start = max(self.query_start, bbox.query_start)
        inter_r_start = max(self.ref_start, bbox.ref_start)
        inter_q_end = min(self.query_end, bbox.query_end)
        inter_r_end = min(self.ref_end, bbox.ref_end)

        # Compute the area of intersection rectangle
        return abs(max((inter_q_end - inter_q_start, 0)) * max((inter_r_end - inter_r_start), 0))

    def overlaps(self, bbox: "Match") -> bool:
        return self.intersection_area(bbox) > 0.


class VideoPair:
    """A video pair item that contains information regarding the gt and pred bboxes.

    Provide functionalities for the combination of new predictions with the
    existing ones and the computation of their intersection with the gt bboxes,
    ignoring the gt bboxes that do not overlap with any prediction.
    """

    def __init__(self, ):
        self.inter_q, self.total_q, self.inter_r, self.total_r = 0, 0, 0, 0
        self.gts, self.preds = [], []

    def total_gt_length(self, ) -> Tuple[int, int]:
        gt_q_ints = Intervals([gt.interval(Axis.QUERY) for gt in self.gts])
        gt_r_ints = Intervals([gt.interval(Axis.REF) for gt in self.gts])
        return gt_q_ints.total_length(), gt_r_ints.total_length()

    def total_pred_length(self, ) -> Tuple[int, int]:
        pred_q_ints = Intervals([pred.interval(Axis.QUERY) for pred in self.preds])
        pred_r_ints = Intervals([pred.interval(Axis.REF) for pred in self.preds])
        return pred_q_ints.total_length(), pred_r_ints.total_length()

    def gt_overlaps(self, gt: Match) -> bool:
        """Checks if the provided gt bbox overlaps with at least one pred bbox."""
        overlaps = False
        for pred in self.preds:
            if gt.overlaps(pred):
                overlaps = True
                break
        return overlaps

    def add_gt(self, bbox: Match):
        self.gts.append(bbox)

    def add_prediction(self, bbox: Match) -> Tuple[float, float, float, float]:
        """Add a prediction to the corresponding list and calculates the
        differences in the intersections with the gt and the total video
        length covered for both query and reference axes.
        """
        self.preds.append(bbox)
        gts_to_consider = [gt for gt in self.gts if self.gt_overlaps(gt)]

        pred_q_ints = Intervals([pred.interval(Axis.QUERY) for pred in self.preds])
        gt_q_ints = Intervals([gt.interval(Axis.QUERY) for gt in gts_to_consider])

        # New intersection and total length on the query axis
        new_inter_q = pred_q_ints.intersect_length(gt_q_ints)
        new_total_q = pred_q_ints.total_length()

        pred_r_ints = Intervals([pred.interval(Axis.REF) for pred in self.preds])
        gt_r_ints = Intervals([gt.interval(Axis.REF) for gt in gts_to_consider])

        # New intersection and total length on the reference axis
        new_inter_r = pred_r_ints.intersect_length(gt_r_ints)
        new_total_r = pred_r_ints.total_length()

        # Compute differences
        diff_inter_q, diff_total_q = new_inter_q - self.inter_q, new_total_q - self.total_q
        diff_inter_r, diff_total_r = new_inter_r - self.inter_r, new_total_r - self.total_r

        # Update with new values
        self.inter_q, self.total_q = new_inter_q, new_total_q
        self.inter_r, self.total_r = new_inter_r, new_total_r

        return diff_inter_q, diff_total_q, diff_inter_r, diff_total_r

    def visualize(self, max_len=30, scale=10, boundary=3):
        """A naive visualization scheme."""
        mask = np.ones((max_len * scale + boundary + 1, max_len * scale + boundary + 1, 3))
        for gt in self.gts:
            qs, qe = gt.query_start * scale, gt.query_end * scale
            rs, re = gt.ref_start * scale, gt.ref_end * scale

            slop = float(qe - qs) / float(re - rs + 1e-7)
            for y in np.arange(qs, qe + boundary, 0.1):
                x = round((y - qs) / slop + rs)
                mask[int(y), int(x), :] = 0.

            mask[qs: qe + boundary + 1, rs] = [1., 0., 0.]
            mask[qs, rs: re + boundary + 1] = [1., 0., 0.]
            mask[qs: qe + boundary + 1, re + boundary] = [1., 0., 0.]
            mask[qe + boundary, rs: re + boundary + 1] = [1., 0., 0.]
        for pr in self.preds:
            qs, qe = pr.query_start * scale, pr.query_end * scale
            rs, re = pr.ref_start * scale, pr.ref_end * scale

            mask[qs: qe + boundary + 1, rs] = [0., 0., 1.]
            mask[qs, rs: re + boundary + 1] = [0., 0., 1.]
            mask[qs: qe + boundary + 1, re + boundary] = [0., 0., 1.]
            mask[qe + boundary, rs: re + boundary + 1] = [0., 0., 1.]

        plt.imshow(mask)
        plt.show()


def match_metric_single_axis(gt: Intervals, preds: Sequence[Tuple[float, float]]):
    """Computes a single-axis matching metric.

    This is equivalent to micro-AP over time units.

    :param gt Ground-truth match intervals.
    :param preds Predictions, ordered by confidence (most confident first).
      Possibly overlapping.
    """
    pred_ints = Intervals()
    gt_length = gt.total_length()
    recall = 0.
    metric = 0.
    for interval in preds:
        pred_ints.add(interval)
        intersect_length = pred_ints.intersect_length(gt)
        new_recall = intersect_length / gt_length
        precision = intersect_length / pred_ints.total_length()
        delta_recall = new_recall - recall
        metric += precision * delta_recall
        recall = new_recall
    return metric


def match_metric_v1(gt: Collection[Match], predictions: Collection[Match]):
    """V1 metric:

    Geometric mean of temporal (1D) uAP across both time axes (query and ref).
    """
    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)
    metrics = []
    for axis in [Axis.QUERY, Axis.REF]:
        gt_int = Intervals([i.interval(axis) for i in gt])
        preds = [i.interval(axis) for i in predictions]
        metrics.append(match_metric_single_axis(gt_int, preds))
    metric = metrics[0] * metrics[1]
    return metrics + [sqrt(metric)]


def match_metric_v2(gts: Collection[Match], predictions: Collection[Match], visualize: bool = False):
    """V2 metric:

    Computes the AP based on the VCSL approach for the
    calculation of Precision and Recall.

    AP = \sum_{i=1}^N P(i) ΔR(i)

    where, P(i) = sqrt(P_q * P_r) and R(i) = sqrt(R_q * R_r)
    calculated as in the VCSL.
    """

    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)

    # Initialize video pairs and load their gt bboxs
    video_pairs = defaultdict(VideoPair)
    for gt in gts:
        pair_id = f"{gt.query_id}-{gt.ref_id}"
        video_pairs[pair_id].add_gt(gt)

    # Get the total gt length for each axis
    gt_q_length = 0
    gt_r_length = 0
    for k, v in video_pairs.items():
        gt_q_len, gt_r_len = v.total_gt_length()
        gt_q_length += gt_q_len
        gt_r_length += gt_r_len

    # Loop through the predictions
    recall, metric = 0., 0.
    inter_q, total_q, inter_r, total_r = 0, 0, 0, 0
    for pred in predictions:
        pair_id = f"{pred.query_id}-{pred.ref_id}"

        # Given a new prediction, we only need the differences in the intersection with
        # gt and total video length covered for both query and reference axes.
        diff_inter_q, diff_total_q, diff_inter_r, diff_total_r = \
            video_pairs[pair_id].add_prediction(pred)

        # Accumulate the differences to the corresponding values
        inter_q += diff_inter_q
        total_q += diff_total_q
        inter_r += diff_inter_r
        total_r += diff_total_r

        # Compute precision and recall
        recall_q = inter_q / gt_q_length
        recall_r = inter_r / gt_r_length

        precision_q = inter_q / total_q
        precision_r = inter_r / total_r

        new_recall = sqrt(recall_q * recall_r)
        precision = sqrt(precision_q * precision_r)

        # Compute metric
        delta_recall = new_recall - recall
        metric += precision * delta_recall
        recall = new_recall

    if visualize:
        for k, v in video_pairs.items():
            v.visualize()
    return metric
