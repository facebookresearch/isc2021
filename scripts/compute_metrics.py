# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import argparse
import matplotlib.pyplot as plt
import numpy as np

from isc.metrics import evaluate, Metrics, PredictedMatch, print_metrics
from isc.io import read_ground_truth, read_predictions, write_predictions, read_descriptors
from isc.descriptor_matching import match_and_make_predictions, knn_match_and_make_predictions
from typing import Optional

import h5py

import faiss



def plot_pr_curve(
    metrics: Metrics, title: str, pr_curve_filepath: Optional[str] = None
):
    _ = plt.figure(figsize=(12, 9))
    plt.plot(metrics.recalls, metrics.precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid()
    if pr_curve_filepath:
        plt.savefig(pr_curve_filepath)




def compute_metrics(args):
    predictions = read_predictions(args.preds_filepath)
    if len(predictions) > args.max_results:
        raise RuntimeError(f"too many predictions ({len(predictions)} > {args.max_results})")

    gt = read_ground_truth(args.gt_filepath)
    print(
        f"Track 1 results of {len(predictions)} predictions ({len(gt)} GT matches)"
    )
    metrics = evaluate(gt, predictions)
    print_metrics(metrics)
    if args.pr_curve_filepath:
        plot_pr_curve(metrics, args.title, args.pr_curve_filepath)
    return metrics



def compute_metrics_track2(args):
    gt = read_ground_truth(args.gt_filepath)



    db_image_ids, xb = read_descriptors(args.db_descs)
    d = xb.shape[1]

    if d > args.max_dim:
        raise RuntimeError(
            f"maximum dimension exceeded {d} > {args.max_dim}"
        )

    query_image_ids, xq = read_descriptors(args.query_descs)

    if d != xq.shape[1]:
        raise RuntimeError(
            f"query descriptors ({xq.shape[1]}) not same dimension as db ({d})"
        )

    if args.knn == -1:
        print(
            f"Track 2 running matching of {len(query_image_ids)} queries in "
            f"{len(db_image_ids)} database ({d}D descriptors), "
            f"max_results={args.max_results}."
        )
        predictions = match_and_make_predictions(
            xq, query_image_ids,
            xb, db_image_ids,
            args.max_results,
            metric = faiss.METRIC_INNER_PRODUCT if args.ip else faiss.METRIC_L2
        )
    else:
        print(
            f"Track 2 running matching of {len(query_image_ids)} queries in "
            f"{len(db_image_ids)} database ({d}D descriptors), "
            f"kNN with k={args.knn}."
        )
        predictions = knn_match_and_make_predictions(
            xq, query_image_ids,
            xb, db_image_ids,
            args.knn,
            metric = faiss.METRIC_INNER_PRODUCT if args.ip else faiss.METRIC_L2
        )

    if args.write_predictions:
        print("writing predictions to", args.write_predictions)
        write_predictions(predictions, args.write_predictions)

    print(f"Evaluating {len(predictions)} predictions ({len(gt)} GT matches)")

    metrics = evaluate(gt, predictions)
    print_metrics(metrics)
    if args.pr_curve_filepath:
        plot_pr_curve(metrics, args.title, args.pr_curve_filepath)
    return metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("input")
    aa("--gt_filepath", required=True, help="CSV file with ground truth")
    aa("--max_results", default=500_000, type=int, help="max number of accepted predictions")
    aa("--knn", default=-1, type=int, help="if not -1, do k-nearest neighbor search with this k")

    group = parser.add_argument_group("for track 1 (matching resutls)")
    aa("--preds_filepath", help="CSV file with predicted matches")

    group = parser.add_argument_group("for track 2 (desciptors)")
    aa("--track2", default=False, action="store_true", help="perform track 2 evaluation (default is track 1)")
    aa("--db_descs", nargs='*', help="database descriptor file in HDF5 format")
    aa("--query_descs", nargs='*', help="query descriptor file in HDF5 format")
    aa("--max_dim", default=256, type=int, help="max number of accepted descriptor dimensions")
    aa("--ip", default=False, action="store_true", help="use inner product rather than L2 to compare descriptors")
    aa("--write_predictions", help="write predictions to this output file (for debugging)")

    group = parser.add_argument_group("output")
    aa("--pr_curve_filepath", default="", help="output file for P-R curve")
    aa("--title", default="", help="title of the plot")

    args = parser.parse_args()

    if not args.track2:
        compute_metrics(args)
    else:
        compute_metrics_track2(args)


