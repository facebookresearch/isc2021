# Running the baseline code of the ISC2021 challenge

This is a step-by-step walkthrough on how to get a valid submission for the
ISC2021 challenge.

This tutorial will guide you to get baseline results for the ISC2021 competion,
how to evaluate them and get a submittable file.

## Downloading the data

Get the images for query, reference and training sets as described in the Driven Data page

https://www.drivendata.org/competitions/80/competition-image-similarity-2-dev/data/

Please be patient, this is a total of 350 GB of data.
Note that the training images are not required for the first steps of the process.

In the following, we assume that the images are available in the `images/queries`, `images/references` and `images/train` subdirectories.

While the data is downloading, you can install the required packages and compile some code.

## Required packages and code

Prepare to run the code with
```
mkdir data/              # where to put the intermediate files
export PYTHONPATH=.      # needed to call the python scripts
```

### Packages to install

python 3.5+

faiss -- GPU will be used if available

typing

h5py

torch + torchvision for multigrain

PIL

### Preparing the GIST extraction

The code for this desctiptor is included in the lear_gist-1.2 subdirectory of the repository.
Try running
```
(cd lear_gist-1.2; make)
```
with a bit of luck it works out-of-the-box.
Otherwise follow the README file to install the required dependency.


## Steps

The tutorial breaks down in steps from easiest and fastest to more complicated.

Step 1: [GIST descriptors on a small subset](1_gist_subset.md)

Step 2: [GIST descriptors with PCA](2_gist_subset_pca.md)

Step 3: [GIST descriptors on the full dataset](2_gist_full.md)

Step 4: [Multigrain descriptors on the small subset](3_multigrain_subset.md)

Step 5: [Multigrain descriptors on the full dataset](3_multigrain_full.md)
