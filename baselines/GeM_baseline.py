# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import sys
import os
import pdb
import argparse
import time
from collections import OrderedDict, defaultdict

from PIL import Image
from isc.io import write_hdf5_descriptors


import torch
import torchvision
import torchvision.transforms
from torch import nn
from torch.utils.data import Dataset

import faiss

import tempfile
import numpy as np
import h5py


def load_model(name, checkpoint_file):
    if name == "zoo_resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        model.eval()
        return model

    if name == "multigrain_resnet50":
        model = torchvision.models.resnet50(pretrained=False)
        st = torch.load(checkpoint_file)
        state_dict = OrderedDict([
            (name[9:], v)
            for name, v in st["model_state"].items() if name.startswith("features.")
        ])
        model.fc
        model.fc = None
        model.load_state_dict(state_dict)
        model.eval()
        return model

    assert False



def resnet_activation_map(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x


def gem_npy(x, p=3, eps=1e-6):
    x = np.clip(x, a_min=eps, a_max=np.inf)
    x = x ** p
    x = x.mean(axis=0)
    return x ** (1. / p)


class ImageList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        x = Image.open(self.image_list[i])
        x = x.convert("RGB")
        if self.imsize is not None:
            x.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        if self.transform is not None:
            x = self.transform(x)
        return x


def main():

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)



    group = parser.add_argument_group('feature extraction options')
    aa('--transpose', default=-1, type=int, help="one of the 7 PIL transpose options ")
    aa('--train_pca', default=False, action="store_true", help="run PCA training")
    aa('--pca_file', default="", help="File with PCA descriptors")
    aa('--pca_dim', default=1500, type=int, help="output dimension for PCA")
    aa('--device', default="cpu", help='pytroch device')
    aa('--batch_size', default=1, type=int, help="max batch size to use for extraction")
    aa('--num_workers', default=20, type=int, help="nb of dataloader workers")

    group = parser.add_argument_group('model options')
    aa('--model', default='multigrain_resnet50', help="model to use")
    aa('--checkpoint', default='/checkpoint/matthijs/HRViT/multigrain_joint_3B_0.5.pth', help='override default checkpoint')
    aa('--GeM_p', default=3.0, type=float, help="Power used for GeM pooling")
    aa('--scales', default="1.0", help="scale levels")
    aa('--imsize', default=800, type=int, help="max image size at extraction time")

    group = parser.add_argument_group('dataset options')
    aa('--file_list', required=True, help="CSV file with image filenames")
    aa('--image_dir', default="", help="search image files in these directories")
    aa('--n_train_pca', default=10000, type=int, help="nb of training vectors for the PCA")
    aa('--i0', default=0, type=int, help="first image to process")
    aa('--i1', default=-1, type=int, help="last image to process + 1")

    group = parser.add_argument_group('output options')
    aa('--o', default="/tmp/desc.hdf5", help="write trained features to this file")

    args = parser.parse_args()
    args.scales = [float(x) for x in args.scales.split(",")]

    print("args=", args)

    print("reading image names from", args.file_list)

    if args.device == "cpu":
        if 'Linux' in platform.platform():
            os.system(
                'echo hardware_image_description: '
                '$( cat /proc/cpuinfo | grep ^"model name" | tail -1 ), '
                '$( cat /proc/cpuinfo | grep ^processor | wc -l ) cores'
            )
        else:
            print("hardware_image_description:", platform.machine(), "nb of threads:", args.nproc)
    else:
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    image_list = [l.strip() for l in open(args.file_list, "r")]

    if args.i1 == -1:
        args.i1 = len(image_list)
    image_list = image_list[args.i0:args.i1]

    # add jpg suffix if there is none
    image_list = [
        fname if "." in fname else fname + ".jpg"
        for fname in image_list
    ]

    # full path name for the image
    image_dir = args.image_dir
    if not image_dir.endswith('/'):
        image_dir += "/"

    image_list = [image_dir + fname for fname in image_list]

    print(f"  found {len(image_list)} images")

    if args.train_pca:
        rs = np.random.RandomState(123)
        image_list = [
            image_list[i]
            for i in rs.choice(len(image_list), size=args.n_train_pca, replace=False)
        ]
        print(f"subsampled {args.n_train_pca} vectors")

    # transform without resizing
    mean, std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]

    transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]

    if args.transpose != -1:
        transforms.insert(TransposeTransform(args.transpose), 0)

    transforms = torchvision.transforms.Compose(transforms)

    im_dataset = ImageList(image_list, transform=transforms, imsize=args.imsize)

    print("loading model")
    net = load_model(args.model, args.checkpoint)
    net.to(args.device)

    print("computing features")

    t0 = time.time()

    with torch.no_grad():
        if args.batch_size == 1:
            all_desc = []
            for no, x in enumerate(im_dataset):
                x = x.to(args.device)
                print(f"im {no}/{len(im_dataset)}    ", end="\r", flush=True)
                x = x.unsqueeze(0)
                feats = []
                for s in args.scales:
                    xs = nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
                    o = resnet_activation_map(net, xs)
                    o = o.cpu().numpy()    # B, C, H, W
                    o = o[0].reshape(o.shape[1], -1).T
                    feats.append(o)

                feats = np.vstack(feats)
                gem = gem_npy(feats, p=args.GeM_p)
                all_desc.append(gem)

        else:
            all_desc = [None] * len(im_dataset)
            ndesc = [0]
            buckets = defaultdict(list)

            def handle_bucket(bucket):
                ndesc[0] += len(bucket)
                x = torch.stack([xi for no, xi in bucket])
                x = x.to(args.device)
                print(f"ndesc {ndesc[0]} / {len(all_desc)} handle bucket of shape {x.shape}\r", end="", flush=True)
                feats = []
                for s in args.scales:
                    xs = nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
                    o = resnet_activation_map(net, xs)
                    o = o.cpu().numpy()    # B, C, H, W
                    feats.append(o)

                for i, (no, _) in enumerate(bucket):
                    feats_i = np.vstack([f[i].reshape(f[i].shape[0], -1).T for f in feats])
                    gem = gem_npy(feats_i, p=args.GeM_p)
                    all_desc[no] = gem

            max_batch_size = args.batch_size

            dataloader = torch.utils.data.DataLoader(
                im_dataset, batch_size=1, shuffle=False,
                num_workers=args.num_workers
            )

            for no, x in enumerate(dataloader):
                x = x[0]  # don't batch
                buckets[x.shape].append((no, x))

                if len(buckets[x.shape]) >= max_batch_size:
                    handle_bucket(buckets[x.shape])
                    del buckets[x.shape]

            for bucket in buckets.values():
                handle_bucket(bucket)

    all_desc = np.vstack(all_desc)

    t1 = time.time()

    print()
    print(f"image_description_time: {(t1 - t0) / len(image_list):.5f} s per image")

    if args.train_pca:
        d = all_desc.shape[1]
        pca = faiss.PCAMatrix(d, args.pca_dim, -0.5)
        print(f"Train PCA {pca.d_in} -> {pca.d_out}")
        pca.train(all_desc)
        print(f"Storing PCA to {args.pca_file}")
        faiss.write_VectorTransform(pca, args.pca_file)
    elif args.pca_file:
        print("Load PCA matrix", args.pca_file)
        pca = faiss.read_VectorTransform(args.pca_file)
        print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
        all_desc = pca.apply_py(all_desc)

    print("normalizing descriptors")
    faiss.normalize_L2(all_desc)

    if not args.train_pca:
        print(f"writing descriptors to {args.o}")
        write_hdf5_descriptors(all_desc, image_list, args.o)


if __name__ == "__main__":
    main()



