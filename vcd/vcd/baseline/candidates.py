from abc import ABC, abstractmethod
from typing import List

import numpy as np
from vcd.index import PairMatches, VideoFeature, VideoIndex
from vcd.metrics import CandidatePair


class ScoreAggregation(ABC):
    @abstractmethod
    def aggregate(self, match: PairMatches) -> float:
        pass

    def score(self, match: PairMatches) -> CandidatePair:
        score = self.aggregate(match)
        return CandidatePair(query_id=match.query_id, ref_id=match.ref_id, score=score)


class MaxScoreAggregation(ScoreAggregation):
    def aggregate(self, match: PairMatches) -> float:
        return np.max([m.score for m in match.matches])


class CandidateGeneration:
    def __init__(self, references: List[VideoFeature], aggregation: ScoreAggregation):
        self.aggregation = aggregation
        dim = references[0].dimensions()
        self.index = VideoIndex(dim)
        self.index.add(references)

    def query(self, queries: List[VideoFeature], k: int = 20) -> List[CandidatePair]:
        matches = self.index.search(queries, k=k)
        candidates = [self.aggregation.score(match) for match in matches]
        candidates = sorted(candidates, key=lambda match: match.score, reverse=True)
        return candidates
