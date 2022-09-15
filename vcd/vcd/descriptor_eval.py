from argparse import ArgumentParser, Namespace

from vcd.baseline.candidates import CandidateGeneration, MaxScoreAggregation
from vcd.metrics import average_precision, CandidatePair, Match
from vcd.storage import load_features

K = 5


def add_eval_args(parser: ArgumentParser):
    parser.add_argument(
        "--query_path", help="Path containing query features", type=str, required=True
    )
    parser.add_argument(
        "--ref_path", help="Path containing reference features", type=str, required=True
    )
    parser.add_argument(
        "--gt_path", help="Path containing Groundtruth", type=str, required=True
    )


parser = ArgumentParser()
add_eval_args(parser)


def main(args: Namespace) -> int:
    print("Starting Descriptor level eval")
    query_features = load_features(args.query_path)
    print(f"Loaded {len(query_features)} query features")
    ref_features = load_features(args.ref_path)
    print(f"Loaded {len(ref_features)} ref features")

    print(f"Performing KNN with k: {K}")
    cg = CandidateGeneration(ref_features, MaxScoreAggregation())
    candidates = cg.query(query_features, k=K)

    gt_matches = Match.read_csv(args.gt_path, is_gt=True)
    gt_pairs = CandidatePair.from_matches(gt_matches)
    print(f"Loaded GT from {args.gt_path}")

    mAP = average_precision(gt_pairs, candidates).ap
    print(f"Micro AP : {mAP}")


if __name__ == "__main__":
    """
    Script to run descriptor level eval and log metrics
    """
    args = parser.parse_args()
    main(args)
