import os
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from vcd.index import VideoFeature


class BaseStorage(ABC):
    """
    Interface to save and load features from storage backend
    """

    def __init__(self):
        pass

    @abstractmethod
    def _save_impl(self, feature: np.ndarray, path: str):
        pass

    def _load_impl(self, path: str) -> np.ndarray:
        pass

    def save(self, features: Dict[str, np.ndarray], path: str):
        os.makedirs(path, exist_ok=True)
        for name, feature in features.items():
            self._save_impl(feature, os.path.join(path, name))

    def load(self, path: str) -> Dict[str, np.ndarray]:
        assert os.path.isdir(
            path
        ), "path: {path}, needs to point to a directory containing features"
        features = {}
        for root, _, files in os.walk(path):
            for f in files:
                features[f] = self._load_impl(os.path.join(root, f))
        return features


class NumpyStorage(BaseStorage):
    def __init__(self):
        super().__init__()

    def _save_impl(self, feature: np.ndarray, path: str):
        with open(path, "wb") as f:
            np.save(f, feature)

    def _load_impl(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            feature = np.load(f)
        return feature


def store_features(f, features: List[VideoFeature]):
    video_ids = []
    feats = []
    timestamps = []
    for feature in features:
        video_ids.append(np.full(len(feature), feature.video_id, dtype=np.int32))
        feats.append(feature.feature)
        timestamps.append(feature.timestamps)
    video_ids = np.concatenate(video_ids)
    feats = np.concatenate(feats)
    timestamps = np.concatenate(timestamps)
    np.savez(f, video_ids=video_ids, features=feats, timestamps=timestamps)


def same_value_ranges(values):
    start = 0
    value = values[start]

    for i, v in enumerate(values):
        if v == value:
            continue
        yield value, start, i
        start = i
        value = values[start]

    yield value, start, len(values)


def load_features(f) -> List[VideoFeature]:
    data = np.load(f, allow_pickle=False)
    video_ids = data["video_ids"]
    feats = data["features"]
    timestamps = data["timestamps"]

    results = []
    for video_id, start, end in same_value_ranges(video_ids):
        results.append(
            VideoFeature(
                video_id=video_id,
                timestamps=timestamps[start:end],
                feature=feats[start:end, :],
            )
        )
    return results
