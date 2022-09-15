import abc
from typing import List

import numpy as np

from vcd.index import VideoFeature
from vcd.metrics import CandidatePair, Match


class Localization(abc.ABC):
    @abc.abstractmethod
    def localize(self, candidate: CandidatePair) -> List[Match]:
        pass

    def localize_all(self, candidates: List[CandidatePair]) -> List[Match]:
        matches = []
        for candidate in candidates:
            matches.extend(self.localize(candidate))
        return matches


class LocalizationWithMetadata(Localization):
    def __init__(self, queries: List[VideoFeature], refs: List[VideoFeature]):
        self.queries = {m.video_id: m for m in queries}
        self.refs = {m.video_id: m for m in refs}

    def similarity(self, candidate: CandidatePair):
        a = self.queries[candidate.query_id].feature
        b = self.refs[candidate.ref_id].feature
        return np.matmul(a, b.T)


class VCSLLocalization(LocalizationWithMetadata):
    def __init__(self, queries, refs, model_type, **kwargs):
        super().__init__(queries, refs)

        # Late import: allow OSS use without VCSL installed
        from vcsl.vta import build_vta_model  # @manual

        self.model = build_vta_model(model_type, **kwargs)

    def localize_all(self, candidates: List[CandidatePair]) -> List[Match]:
        sims = [(f"{c.query_id}-{c.ref_id}", self.similarity(c)) for c in candidates]
        results = self.model.forward_sim(sims)
        assert len(results) == len(candidates)
        matches = []
        for (candidate, (key, sim), result) in zip(candidates, sims, results):
            q_ts = self.queries[candidate.query_id].timestamps
            r_ts = self.refs[candidate.ref_id].timestamps
            assert key == result[0]
            for box in result[1]:
                (x1, y1, x2, y2) = box
                match = Match(
                    query_id=candidate.query_id,
                    ref_id=candidate.ref_id,
                    query_start=q_ts[x1],
                    query_end=q_ts[x2],
                    ref_start=r_ts[y1],
                    ref_end=r_ts[y2],
                )
                score = self.score(candidate, match, box, sim)
                match = match._replace(score=score)
                matches.append(match)
        return matches

    def localize(self, candidate: CandidatePair) -> List[Match]:
        return self.localize_all([candidate])

    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        return 1.0


class VCSLLocalizationMaxSim(VCSLLocalization):
    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        x1, y1, x2, y2 = box
        return similarity[x1:x2, y1:y2].max()


class VCSLLocalizationCandidateScore(VCSLLocalization):
    def score(self, candidate: CandidatePair, match: Match, box, similarity) -> float:
        return candidate.score
