# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import argparse
import matplotlib.pyplot as plt
import numpy as np
import faiss

import faiss.contrib

from isc.descriptor_matching import match_and_make_predictions
from isc.io import write_predictions, read_descriptors

import faiss
import logging
import gc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("input")
    aa("--db_descs", nargs='*', help="database descriptor file in HDF5 format")
    aa("--train_descs", nargs='*', help="training descriptor file in HDF5 format")
    aa("--query_descs", nargs='*', help="query descriptor file in HDF5 format")

    group = parser.add_argument_group("normalization parameters")
    aa("--n", default=-1, type=int, help="rank in training results to use for normalization")
    aa("--factor", default=1.0, type=float, help="weight of the normalization")
    aa("--reduction", default="min",
        choices=["min", "avg"],
        help="how to use the result list to compute the normalization")
    aa("--read_norms", default=False, action="store_true",
        help="use the norms vector without computing it")

    group = parser.add_argument_group("output")
    aa("--o", default="/tmp/preds.csv", help="write predictions to this output file")
    aa("--max_results", default=500_000, type=int, help="max number of accepted predictions")
    aa("--norms", help="write the nromalization factors (for debugging)")

    args = parser.parse_args()
    print("args=", args)

    print("loading query descriptors")
    query_image_ids, xq = read_descriptors(args.query_descs)
    print("loading db descriptors")
    db_image_ids, xb = read_descriptors(args.db_descs)

    d = xq.shape[1]
    if args.read_norms:
        norms = np.load(args.norms)
    else:
        print("loading train descriptors")

        train_image_ids, xt = read_descriptors(args.train_descs)

        print(
            f"Matching {len(xq)} queries in {len(xt)} training ({d}D descriptors), "
            f"keeping n={args.n} neighbors per query."
        )

        # We use inner-product scoring, which is equivalent to L2 for normalized features
        # but easier to manipulate.
        index = faiss.IndexFlatIP(d)
        index.add(xt)
        if faiss.get_num_gpus() > 0:
            print(f"running on {faiss.get_num_gpus()} GPUs")
            index = faiss.index_cpu_to_all_gpus(index)

        train_scores, I = index.search(xq, args.n)

        print(f"matching scores in [{train_scores.min():g}, {train_scores.max():g}]")
        print(f"computing normalization type {args.reduction} with factor {args.factor}")

        if args.reduction == "min":
            norms = -train_scores[:, -1] * args.factor
        elif args.reduction == "mean":
            norms = -train_scores.mean(axis=1) * args.factor
        else:
            assert False

        if args.norms:
            print("writing normalizations to", args.norms)
            np.save(args.norms, norms)

        del index; gc.collect()

    print(f"   normalization range [{norms.min():g}, {norms.max():g}]")

    # Add one column to the matrices so that the inner-product computes
    # normalized matches.

    xq_1 = np.hstack((xq, norms[:, None]))
    xb_1 = np.hstack((xb, np.ones((len(xb), 1), dtype='float32')))

    predictions = match_and_make_predictions(
            xq_1, query_image_ids,
            xb_1, db_image_ids,
            args.max_results,
            metric=faiss.METRIC_INNER_PRODUCT
    )

    print("writing predictions to", args.o)
    write_predictions(predictions, args.o)

