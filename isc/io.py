# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

from typing import Iterable, List, Optional

import numpy as np
import h5py

from .metrics import GroundTruthMatch, PredictedMatch


def read_ground_truth(filename: str) -> List[GroundTruthMatch]:
    """
    Read groundtruth csv file.
    Must contain query_image_id,db_image_id on each line.
    handles the no header version and DD's version with header
    """
    gt_pairs = []
    with open(filename, "r") as cfile:
        for line in cfile:
            line = line.strip()
            if line == 'query_id,reference_id':
                continue
            q, db = line.split(",")
            if db == '':
                continue
            gt_pairs.append(GroundTruthMatch(q, db))
    return gt_pairs


def read_predictions(filename: str) -> List[PredictedMatch]:
    """
    Read predictions csv file.
    Must contain query_image_id,db_image_id,score on each line.
    Header optional
    """
    predictions = []
    with open(filename, "r") as cfile:
        for line in cfile:
            line = line.strip()
            if line == "query_id,reference_id,score":
                continue
            q, db, score = line.split(",")
            predictions.append(PredictedMatch(q, db, float(score)))
    return predictions


def write_predictions(
    predictions: Iterable[PredictedMatch],
    preds_filepath: str,
):
    with open(preds_filepath, "w") as pfile:
        pfile.write("query_id,reference_id,score\n")
        for p in predictions:
            row = f"{p.query},{p.db},{p.score:.6f}"
            pfile.write(row + "\n")


def write_predictions_from_arrays(
    S: np.ndarray,
    I: np.ndarray,
    dbids: np.ndarray,
    qids: np.ndarray,
    preds_filepath: str,
    nmax: Optional[int] = None,
    score_min: Optional[float] = None,
):
    """
    Write CSV predictions file from arrays returned by FAISS.


    Parameters
    ----------
    S : np.ndarray
        Scores for each query. Shape nq, k.
        For each query, scores must be in decreasing order.
    I : np.ndarray
        Nearest neighbors indices for each query. Shape nq, k
    dbids : np.ndarray
        Image ids for each reference image. Shape [nb, ]
    qids : np.ndarray
        Image ids for each query image. Shape [nq, ]
    preds_filepath : str
        Output file path.
    nmax : Optional[int], optional
        Maximum number of predictions to write. Will pick the ones with highest score.
        If None, no limit on the number of predictions.
    score_min : Optional[float], optional
        Minimum score to write a predictions.
        If None, no limit.
    """
    nq, k = S.shape
    # Find the minimum score to return at most nmax predictions
    # TODO(lowik) might not work if duplicates scores at nmax index...
    if nmax is None or nq * k <= nmax:
        score_min = -1e6
    else:
        S = S.copy()   # we are going to overwrite it
        scores = S.ravel()  # linear view of array
        pivot = len(scores) - nmax
        o = scores.argpartition(pivot)  #
        scores[o[:pivot]] = -1e7
        score_min = -1e6

    with open(preds_filepath, "w") as pfile:
        for qidx in range(nq):
            query_id = qids[qidx]
            for score, dbidx in zip(S[qidx], I[qidx]):
                db_id = dbids[dbidx]
                if score >= score_min:
                    row = f"{query_id},{db_id},{score:.6f}"
                    pfile.write(row + "\n")
                else:
                    # Assume scores are in decreasing order in the array
                    # next scores on this row won't pass the threshold
                    break


def write_predictions_from_range_arrays(
    lims: np.ndarray,
    S: np.ndarray,
    I: np.ndarray,
    dbids: np.ndarray,
    qids: np.ndarray,
    preds_filepath: str,
    nmax: Optional[int] = None,
):
    """
    Write CSV predictions file from range search arrays returned by FAISS.

    Parameters
    ----------
    lims : np.ndarray
        limits between queries, shape (nq + 1)
    S : np.ndarray
        Flat array of scores
    I : np.ndarray
        Flat array of indices
    dbids : np.ndarray
        Image ids for each reference image. Shape [nb, ]
    qids : np.ndarray
        Image ids for each query image. Shape [nq, ]
    preds_filepath : str
        Output file path.
    nmax : Optional[int], optional
        Maximum number of predictions to write. Will pick the ones with highest score.
        If None, no limit on the number of predictions.
    """
    npred, = S.shape
    nq = len(lims) - 1
    assert lims[-1] == npred
    assert I.shape == (npred, )

    if nmax is None or S.size <= nmax:
        score_min = -1e6
    else:
        S = S.copy()   # we are going to overwrite it
        scores = S.ravel()
        pivot = len(scores) - nmax
        o = scores.argpartition(pivot)  #
        scores[o[:pivot]] = -1e7
        score_min = -1e6

    with open(preds_filepath, "w") as pfile:
        for qidx in range(nq):
            l0, l1 = lims[qidx:qidx + 2]
            query_id = qids[qidx]
            for dbidx in range(l0, l1):
                db_id = dbids[I[dbidx]]
                score = S[dbidx]
                if score > score_min:
                    row = f"{query_id},{db_id},{score:.6f}"
                    pfile.write(row + "\n")


def write_hdf5_descriptors(vectors, image_names, fname):
    """
    write image description vectors in HDF5 format.
    """
    # image_names = np.array(image_names)
    vectors = np.ascontiguousarray(vectors, dtype='float32')
    image_names = np.array([
        bytes(name, "ascii")
        for name in image_names
    ])
    with h5py.File(fname, "w") as f:
        f.create_dataset("vectors", data=vectors)
        f.create_dataset("image_names", data=image_names)


def read_descriptors(filenames):
    """ read descriptors from a set of HDF5 files """
    descs = []
    names = []
    for filename in filenames:
        hh = h5py.File(filename, "r")
        descs.append(np.array(hh["vectors"]))
        names += np.array(hh["image_names"][:], dtype=object).astype(str).tolist()
    # strip paths and extensions from the filenames
    names = [
        name.split('/')[-1]
        for name in names
    ]
    names = [
        name[:-4] if name.endswith(".jpg") or name.endswith(".png") else name
        for name in names
    ]
    return names, np.vstack(descs)

