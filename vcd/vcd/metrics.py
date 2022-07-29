import enum
from math import sqrt
from typing import NamedTuple, Collection, List, Optional, Sequence, Tuple


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

    def interval(self, axis: Axis) -> Tuple[float, float]:
        if axis == Axis.QUERY:
            return (self.query_start, self.query_end)
        else:
            return (self.ref_start, self.ref_end)


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


def matching_metric(gt: Collection[Match], predictions: Collection[Match]):
    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)
    metrics = []
    for axis in [Axis.QUERY, Axis.REF]:
        gt_int = Intervals([i.interval(axis) for i in gt])
        preds = [i.interval(axis) for i in predictions]
        metrics.append(match_metric_single_axis(gt_int, preds))
    metric = metrics[0] * metrics[1]
    return metrics + [sqrt(metric)]



