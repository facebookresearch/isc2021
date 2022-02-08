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

**Update (2022-02-08):**  After the competition, the data is available at: https://sites.google.com/view/isc2021/dataset

In the following, we assume that the images are available in the `images/queries`, `images/references` and `images/train` subdirectories.

While the data is downloading, you can install the required packages and compile some code.

## Cloning & installing dependencies

First, clone this repo:
```bash
git clone https://github.com/facebookresearch/isc2021.git
```

Follow the steps below to install all the required dependencies in order to run the ISC evaluation code. Note: The code in this repo requires 3.5 <= Python <= 3.8.

```bash
conda create -n isc2021 python=3.8 -y && conda activate isc2021
pip install -e isc2021/
conda install -c pytorch faiss-gpu
```

## Steps

The tutorial breaks down in steps from easiest and fastest to more complicated.

Step 1: [GIST descriptors on a small subset](1_gist_subset.md)

Step 2: [GIST descriptors with PCA](2_gist_subset_pca.md)

Step 3: [GIST descriptors on the full dataset](3_gist_full.md)

Step 4: [Multigrain descriptors on the small subset](4_multigrain_subset.md)

Step 5: [Multigrain descriptors on the full dataset](5_multigrain_full.md)

Step 6: [HOW local features on the small subset](6_how_subset.md)

Step 7: [HOW local features on the full dataset](7_how_full.md)
