# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import argparse
import numpy as np

import h5py

from isc.io import read_descriptors

def sort_descriptors(ids, x):
    # convert to bytes as hdf5 does not always handle unicode well
    ids = np.array([
        bytes(name, "ascii")
        for name in ids
    ])
    o = ids.argsort()
    return ids[o], x[o].astype("float32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group("track 2 input")
    aa("--db_descs", nargs='*', help="database descriptor file in HDF5 format")
    aa("--query_descs", nargs='*', help="query descriptor file in HDF5 format")

    group = parser.add_argument_group("output")
    aa("--o", default="/tmp/sub.hdf5", help="output in official format for all descriptors")

    args = parser.parse_args()

    db_image_ids, xb = read_descriptors(args.db_descs)
    d = xb.shape[1]

    query_image_ids, xq = read_descriptors(args.query_descs)

    if d != xq.shape[1]:
        raise RuntimeError(
            f"query descriptors ({xq.shape[1]}) not same dimension as db ({d})"
        )

    if d > 256:
        print(f"warning, dimension {d} is larger than allowed for track 2")

    # sort the descriptors (the official eval expects them sorted)
    query_image_ids, xq = sort_descriptors(query_image_ids, xq)
    db_image_ids, xb = sort_descriptors(db_image_ids, xb)

    print("writing", args.o)

    with h5py.File(args.o, "w") as f:
        f.create_dataset("query", data=xq)
        f.create_dataset("reference", data=xb)
        f.create_dataset('query_ids', data=query_image_ids)
        f.create_dataset('reference_ids', data=db_image_ids)


