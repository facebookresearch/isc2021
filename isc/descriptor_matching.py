# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import numpy as np
import faiss
from faiss.contrib import exhaustive_search

from .metrics import PredictedMatch

def query_iterator(xq):
    """ produces batches of progressively increasing sizes """
    nq = len(xq)
    bs = 32
    i = 0
    while i < nq:
        xqi = xq[i : i + bs]
        yield xqi
        if bs < 20000:
            bs *= 2
        i += len(xqi)

def search_with_capped_res(xq, xb, num_results):
    """
    Searches xq into xb, with a maximum total number of results
    """
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    # logging.basicConfig()
    # logging.getLogger(exhaustive_search.__name__).setLevel(logging.DEBUG)

    radius, lims, dis, ids = exhaustive_search.range_search_max_results(
        index, query_iterator(xq),
        1e10,      # initial radius is arbitrary
        max_results=2 * num_results,
        min_results=num_results,
        ngpu=0    # use GPU if available
    )

    n = len(dis)
    nq = len(xq)
    if n > num_results:
        # crop to num_results exactly
        o = dis.argpartition(num_results)[:num_results]
        mask = np.zeros(n, bool)
        mask[o] = True
        new_dis = dis[mask]
        new_ids = ids[mask]
        nres = [0] + [
            mask[lims[i] : lims[i + 1]].sum()
            for i in range(nq)
        ]
        new_lims = np.cumsum(nres)
        lims, dis, ids = new_lims, new_dis, new_ids

    return lims, dis, ids


def match_and_make_predictions(xq, query_image_ids, xb, db_image_ids, num_results, ngpu=-1):
    lims, dis, ids = search_with_capped_res(xq, xb, num_results)
    nq = len(xq)

    predictions = [
        PredictedMatch(
            query_image_ids[i],
            db_image_ids[ids[j]],
            -dis[j]
        )
        for i in range(nq)
        for j in range(lims[i], lims[i + 1])
    ]
    return predictions


def knn_match_and_make_predictions(xq, query_image_ids, xb, db_image_ids, k, ngpu=-1):

    if faiss.get_num_gpus() == 0 or ngpu == 0:
        D, I = faiss.knn(xq, xb, k)
    else:
        d = xq.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        index = faiss.index_cpu_to_all_gpus(index)
        D, I = index.search(xq, k=k)
    nq = len(xq)

    predictions = [
        PredictedMatch(
            query_image_ids[i],
            db_image_ids[I[i, j]],
            -D[i, j]
        )
        for i in range(nq)
        for j in range(k)
    ]
    return predictions




def range_result_read(fname):
    """ read the range search result file format """
    f = open(fname, "rb")
    nq, total_res = np.fromfile(f, count=2, dtype="int32")
    nres = np.fromfile(f, count=nq, dtype="int32")
    assert nres.sum() == total_res
    I = np.fromfile(f, count=total_res, dtype="int32")
    return nres, I