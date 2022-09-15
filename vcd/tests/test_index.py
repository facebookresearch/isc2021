import unittest

import faiss  # @manual

import numpy as np

from vcd.index import VideoFeature, VideoIndex


class IndexTest(unittest.TestCase):
    def test_video_index(self):
        # test_feature = np.random.rand(50, 50, 32)
        test_feature = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
                [[111, 112, 113], [114, 115, 116], [117, 118, 119]],
            ],
            dtype=np.float32,
        )
        query = [
            VideoFeature(
                video_id=idx, feature=feature, timestamps=np.arange(3, dtype=np.float32)
            )
            for idx, feature in enumerate(test_feature)
        ]
        db = [
            VideoFeature(
                video_id=idx, feature=feature, timestamps=np.arange(3, dtype=np.float32)
            )
            for idx, feature in enumerate(test_feature)
        ]

        index = VideoIndex(3, "Flat", faiss.METRIC_L1)
        index.add(db)
        results = index.search(query, 1)
        for result in results:
            self.assertEqual(result.query_id, result.ref_id)
