import collections
from dataclasses import dataclass
from typing import List, NamedTuple

import faiss  # @manual
import numpy as np


@dataclass
class VideoMetadata:
    video_id: int
    timestamps: np.ndarray

    def __len__(self):
        return len(self.timestamps)


@dataclass
class VideoFeature(VideoMetadata):
    feature: np.ndarray

    def metadata(self):
        return VideoMetadata(video_id=self.video_id, timestamps=self.timestamps)

    def dimensions(self):
        return self.feature.shape[1]


class PairMatch(NamedTuple):
    query_timestamp: float
    ref_timestamp: float
    score: float


@dataclass
class PairMatches:
    query_id: int
    ref_id: int
    matches: List[PairMatch]

    def records(self):
        for match in self.matches:
            yield {
                "query_id": self.query_id,
                "ref_id": self.ref_id,
                "query_ts": match.query_timestamp,
                "ref_ts": match.ref_timestamp,
                "score": match.score,
            }


class VideoIndex:
    def __init__(
        self,
        dim: int,
        codec_str: str = "Flat",
        metric: int = faiss.METRIC_INNER_PRODUCT,
    ):
        self.dim = dim
        self.index = faiss.index_factory(self.dim, codec_str, metric)
        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        self.video_clip_idx = []
        self.video_clip_to_video_idx = []
        self.video_metadata = {}

    def add(self, db: List[VideoFeature]):
        for vf in db:
            self.video_clip_idx.extend(list(range(vf.feature.shape[0])))
            self.video_clip_to_video_idx.extend(
                [vf.video_id for _ in range(vf.feature.shape[0])]
            )
            self.video_metadata[vf.video_id] = vf.metadata()
            self.index.add(vf.feature)

    def search(self, queries: List[VideoFeature], k: int = 20) -> List[PairMatches]:
        query_ids = []
        query_indices = []
        for q in queries:
            query_ids.extend([q.video_id] * len(q))
            query_indices.extend(range(len(q)))
        query_metadatas = {q.video_id: q.metadata() for q in queries}
        query_features = np.concatenate([q.feature for q in queries])
        D, I = self.index.search(query_features, k)

        pair_nns = collections.defaultdict(list)

        for i in range(D.shape[0]):
            query_id = query_ids[i]
            query_idx = query_indices[i]
            query_metadata = query_metadatas[query_id]
            ref_ids = [self.video_clip_to_video_idx[i] for i in I[i]]
            ref_indices = [self.video_clip_idx[i] for i in I[i]]
            query_timestamp = query_metadata.timestamps[query_idx]
            for ref_id, ref_idx, score in zip(ref_ids, ref_indices, D[i]):
                ref_metadata = self.video_metadata[ref_id]
                match = PairMatch(
                    query_timestamp=query_timestamp,
                    ref_timestamp=ref_metadata.timestamps[ref_idx],
                    score=score,
                )
                pair_nns[query_id, ref_id].append(match)

        return [
            PairMatches(query_id, ref_id, matches)
            for ((query_id, ref_id), matches) in pair_nns.items()
        ]
