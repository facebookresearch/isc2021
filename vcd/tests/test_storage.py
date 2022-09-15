import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from vcd.index import VideoFeature
from vcd.storage import load_features, store_features


class SorageTest(unittest.TestCase):
    dims = 32

    def fake_vf(self, video_id, length, fps=1.0):
        embeddings = np.random.randn(length, self.dims)
        timestamps = np.arange(length) / fps
        return VideoFeature(
            video_id=video_id, timestamps=timestamps, feature=embeddings
        )

    def test_merged_storage(self):
        features = [
            self.fake_vf(2, 10),
            self.fake_vf(3, 20, fps=3.0),
            self.fake_vf(1, 30, fps=0.5),
        ]
        with tempfile.NamedTemporaryFile() as f:
            store_features(f, features)
            f.flush()
            restored = load_features(f.name)

        self.assertEqual(len(features), len(restored))
        for a, b in zip(features, restored):
            self.assertEqual(a.video_id, b.video_id)
            assert_allclose(b.timestamps, a.timestamps)
            assert_allclose(b.feature, a.feature)
