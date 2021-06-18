# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import sys
import os
import argparse
import pickle
import subprocess
import platform
import time
from multiprocessing import Pool as ProcessPool

from PIL import Image
from isc.io import write_hdf5_descriptors

import faiss

import tempfile
import numpy as np
import h5py

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

class GISTFeatures:
    """ calling GIST executable on a stream of images"""

    def __init__(self, execname, transpose=-1):
        self.name = "gist"
        self.imsz = 32  # compute on images resized to 64*64 pix
        self.d = 960
        self.transpose = transpose
        self.execname = execname

    def preproc_image(self, im):
        """ resize to imsz in the largest dimension and crop out central part """
        w, h = im.size
        if w > h:
            im = im.resize((self.imsz * w // h, self.imsz), Image.BILINEAR)
            w2 = im.size[0]
            pad = (w2 - self.imsz) / 2
            im = im.crop((pad, 0, pad + self.imsz, self.imsz))
        else:
            im = im.resize((self.imsz * h // w, self.imsz), Image.BILINEAR)
            h2 = im.size[1]
            pad = (h2 - self.imsz) / 2
            im = im.crop((0, pad, self.imsz, pad + self.imsz))
        if self.transpose != -1:
            im = im.transpose(self.transpose)
        assert im.size == (self.imsz, self.imsz)
        return im

    def extract_from_image_list(self, imlist, outfile):
        execfile = self.execname
        assert os.path.exists(execfile), (
            "executable %s does not exist, " % execfile
            + "please do: make in the excutable directory"
        )

        f = subprocess.Popen(
            [execfile, "-o", outfile],
            text=False,
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL
        )
        for i, imname in enumerate(imlist):
            if i < 100 or i % 100 == 0:
                print(f"processing {i}/{len(imlist)} {imname}", end="\r", flush=True)
            im = Image.open(imname)
            if im.mode != "RGB":
                im = im.convert('RGB')
            im = self.preproc_image(im)
            im.save(f.stdin, format="ppm")
        f.stdin.close()
        f.wait()


    def compute_features(self, imlist):
        try:
            _, tmpname = tempfile.mkstemp()
            self.extract_from_image_list(imlist, tmpname)
            return fvecs_read(tmpname)
        finally:
            if os.path.exists(tmpname):
                os.unlink(tmpname)

def main():

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('runtime options')
    aa('--nproc', default=0, type=int, help="number of subprocesses to use")
    aa('--giststream_exec',
        default="lear_gist-1.2/compute_gist_stream",
        help="executable that extracts GIST features")

    group = parser.add_argument_group('feature extraction options')
    aa('--transpose', default=-1, type=int, help="one of the 7 PIL transpose options ")
    aa('--train_pca', default=False, action="store_true", help="run PCA training")
    aa('--pca_file', default="", help="File with PCA descriptors")
    aa('--pca_dim', default=256, type=int, help="output dimension for PCA")
    aa('--pca_white', default=0.0, type=float, help="set to -0.5 to whiten PCA")

    group = parser.add_argument_group('dataset options')
    aa('--file_list', required=True, help="CSV file with image filenames")
    aa('--image_dir', default="", help="search image files in this directory")
    aa('--n_train_pca', default=10000, type=int, help="nb of training vectors for the PCA")
    aa('--i0', default=0, type=int, help="first image to process")
    aa('--i1', default=-1, type=int, help="last image to process + 1")

    group = parser.add_argument_group('output options')
    aa('--o', default="/tmp/desc.hdf5", help="write trained features to this file")

    args = parser.parse_args()
    print("args=", args)

    print("reading image names from", args.file_list)

    if 'Linux' in platform.platform():
        os.system(
            'echo hardware_image_description: '
            '$( cat /proc/cpuinfo | grep ^"model name" | tail -1 ), '
            '$( cat /proc/cpuinfo | grep ^processor | wc -l ) cores'
        )
    else:
        print("hardware_image_description:", platform.machine(), "nb of threads:", args.nproc)

    image_ids = [l.strip() for l in open(args.file_list, "r")]

    if args.i1 == -1:
        args.i1 = len(image_ids)
    image_ids = image_ids[args.i0:args.i1]

    # full path name for the image
    image_dir = args.image_dir
    if not image_dir.endswith('/'):
        image_dir += "/"

    # add jpg suffix if there is none
    image_list = [
        image_dir + fname if "." in fname else image_dir + fname + ".jpg"
        for fname in image_ids
    ]

    print(f"  found {len(image_list)} images")

    if args.train_pca:
        rs = np.random.RandomState(123)
        image_list = [
            image_list[i]
            for i in rs.choice(len(image_list), size=args.n_train_pca, replace=False)
        ]
        print(f"subsampled {args.n_train_pca} vectors")

    print("computing features")

    gist = GISTFeatures(
        execname=args.giststream_exec,
        transpose=args.transpose
    )

    t0 = time.time()

    if args.nproc == 0:
        all_desc = gist.compute_features(image_list)
    else:
        pool = ProcessPool(args.nproc)
        n = len(image_list)
        nproc = args.nproc
        sub_lists = [
            image_list[i * n // nproc : (i + 1) * n // nproc]
            for i in range(nproc)
        ]
        all_desc = list(pool.map(gist.compute_features, sub_lists))
        all_desc = np.vstack(all_desc)

    # normalization is important
    faiss.normalize_L2(all_desc)

    t1 = time.time()
    print()
    print(f"image_description_time: {(t1 - t0) / len(image_list):.5f} s per image")

    if args.train_pca:
        d = all_desc.shape[1]
        pca = faiss.PCAMatrix(d, args.pca_dim, args.pca_white)
        print(f"Train PCA {pca.d_in} -> {pca.d_out}")
        pca.train(all_desc)
        print(f"Storing PCA to {args.pca_file}")
        faiss.write_VectorTransform(pca, args.pca_file)
    elif args.pca_file:
        print("Load PCA matrix", args.pca_file)
        pca = faiss.read_VectorTransform(args.pca_file)
        print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
        all_desc = pca.apply_py(all_desc)


    if not args.train_pca:
        print(f"writing descriptors to {args.o}")
        write_hdf5_descriptors(all_desc, image_ids, args.o)


if __name__ == "__main__":
    main()



