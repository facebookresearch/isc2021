"""
Find best image-pair matches from HOW method results
"""

import sys
import argparse
import pickle
from pathlib import Path
import numpy as np

from how.utils import io_helpers

HOW_ROOT = Path("baselines/how")
MAX_RESULTS = 500_000


def find_matches(args):
    """Given results files, create csv predictions file"""
    databases = [args.database]
    if args.normalization_set:
        databases.append(args.normalization_set)

    # Load list files
    queries = load_list_file(args.query_list)
    references = load_list_file(args.db_list)
    # Load scenario parameters
    params = io_helpers.load_params(args.scenario)
    exp_name = args.scenario.rsplit("/", 1)[1][:-len(args.scenario.rsplit(".", 1)[1])-1]
    eval_folder = HOW_ROOT / params['demo_eval']['exp_folder'] / exp_name / "eval"
    # Load results for both databases
    results = {}
    for database in databases:
        results[database] = load_results(eval_folder.glob(f"{database}.results*.pkl"))
    if args.normalization_set:
        assert (results[args.normalization_set]['query_ids'] == results[args.database]['query_ids']).all()

    # Normalize references scores by train scores
    ranks = results[args.database]['ranks']
    scores = results[args.database]['scores']
    if args.normalization_set:
        norm_reduction, norm_rank, norm_factor = "min", 9, 2
        if norm_reduction == "min":
            norms = results[args.normalization_set]['scores'][:,norm_rank]
        elif norm_reduction == "mean":
            norms = results[args.normalization_set]['scores'][:,:norm_rank+1].mean(axis=1)
        scores -= norm_factor * norms[:,None]
    # Take top predictions
    top_idxs = np.argsort(-scores.flatten())[:MAX_RESULTS]
    idx_query, idx_rank = np.unravel_index(top_idxs, scores.shape)
    idx_db, score = ranks[idx_query,idx_rank], scores[idx_query,idx_rank]
    predictions = [(queries[q], references[db], s) for q, db, s in zip(idx_query, idx_db, score)]
    if args.preds_filepath:
        store_predictions(predictions, args.preds_filepath)


def load_list_file(path):
    """Load an image list from a text file"""
    names = []
    with open(path, 'r') as handle:
        for line in handle:
            names.append(line.strip())
    return names

def load_results(paths):
    """Load search results from a list of pickle files"""
    paths = list(paths)
    if not paths:
        raise OSError("No results path found")
    results = []
    for path in sorted(paths):
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
            results.append((data['query_ids'], data['ranks'], data['scores']))

    results = tuple(zip(*results))
    query_ids, ranks, scores = np.hstack(results[0]), np.vstack(results[1]), np.vstack(results[2])
    assert len(query_ids) == len(ranks) == len(scores)
    return {"query_ids": query_ids, "ranks": ranks, "scores": scores}

def store_predictions(predictions, output_path):
    """Store predictions in the csv format"""
    with open(output_path, 'w') as handle:
        for query, reference, score in predictions:
            handle.write(f"{query},{reference},{score}\n")


def main(args):
    """Command-line entrypoint"""
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("database")
    parser.add_argument("--normalization_set")
    parser.add_argument("--query_list")
    parser.add_argument("--db_list")
    parser.add_argument("--preds_filepath")
    args = parser.parse_args(args)

    find_matches(args)


if __name__ == "__main__":
    main(sys.argv[1:])
