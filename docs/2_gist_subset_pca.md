# GIST with PCA descriptor

In this step, we will reduce the dimension of the descriptors
so that the result can be submitted as a real track 2 result.

## Training the PCA

For this we will need the training set. We will use a subset of just 4000
images for training.

Note that the PCA cannot be trained on the reference images because that
would mean that the scoring of a (query, database) pair would depend on
other reference images, which is prohibited by the rules.

The following command extracts GIST features from 4000 images
(sampled randomly from the training set) and trains a PCA to 256 dimensions from these:
```bash
python baselines/gist_baseline.py  \
      --pca_dim 256 \
      --file_list list_files/train \
      --image_dir images/train \
      --nproc 20 \
      --pca_file data/pca_gist.vt \
      --n_train_pca 4000 \
      --train_pca
```

## Running the feature extraction

The feature extraction works as before, except that we apply a PCA
dimensionality reduction:
```bash
python baselines/gist_baseline.py \
    --file_list list_files/subset_1_queries \
    --image_dir images/queries \
    --o data/subset_1_queries_gist_pca.hdf5 \
    --pca_file data/pca_gist.vt \
    --nproc 20

python baselines/gist_baseline.py \
    --file_list list_files/subset_1_references \
    --image_dir images/references \
    --o data/subset_1_references_gist_pca.hdf5 \
    --pca_file data/pca_gist.vt \
    --nproc 20
```

## Evaluating

The evaluation works as before:
```bash
python scripts/compute_metrics.py \
    --query_descs data/subset_1_queries_gist_pca.hdf5 \
    --db_descs data/subset_1_references_gist_pca.hdf5 \
    --gt_filepath list_files/subset_1_ground_truth.csv \
    --track2
```
Note that this time, it is not necessary to override the
allowed descriptor dimension.

```
Track 2 running matching of 4991 queries in 4991 database (256D descriptors), max_results=500000.
Evaluating 500000 predictions (4991 GT matches)
Average Precision: 0.27021
Recall at P90    : 0.23062
Threshold at P90 : -0.130821
Recall at rank 1:  0.32899
Recall at rank 10: 0.37588
```

The average precision is slightly better than in full dimension:
PCA dimensionality reduction is both more compact and more accurate!