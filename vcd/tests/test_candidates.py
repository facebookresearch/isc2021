import unittest

import numpy as np

from vcd.baseline.candidates import CandidateGeneration, MaxScoreAggregation
from vcd.index import VideoFeature
from vcd.metrics import CandidatePair


class CandidateGenerationTest(unittest.TestCase):
    def test_candidate_generation(self):
        queries = [
            VideoFeature(
                video_id=1,
                feature=np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                ),
                timestamps=[0.0, 1.0, 2.0],
            ),
        ]
        refs = [
            VideoFeature(
                video_id=5,
                feature=np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 2, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
                timestamps=[2.0, 4.0, 6.0, 8.0],
            ),
            VideoFeature(
                video_id=8,
                feature=np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                    dtype=np.float32,
                ),
                timestamps=[0.0, 5.0, 10.0],
            ),
            VideoFeature(
                video_id=10,
                feature=np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0.25],
                        [0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
                timestamps=[0.0, 0.1, 0.2],
            ),
        ]

        cg = CandidateGeneration(refs, MaxScoreAggregation())
        candidates = cg.query(queries, k=2)

        self.assertEqual(3, len(candidates))
        self.assertEqual(
            candidates,
            [
                CandidatePair(query_id=1, ref_id=5, score=2.0),
                CandidatePair(query_id=1, ref_id=8, score=1.0),
                CandidatePair(query_id=1, ref_id=10, score=0.25),
            ],
        )


if __name__ == "__main__":
    unittest.main()
