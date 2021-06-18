
# Extracting MultiGrain features on a small dataset

## MultiGrain

The MultiGrain features are produced by inference from a Resnet50 model that
was trained in part to resist data augmentation.
For more details, see the paper [MultiGrain: a unified image embedding for classes and instances](https://arxiv.org/abs/1902.05509), by Berman et al. or the [MultiGrain github repository](https://github.com/facebookresearch/multigrain).

For this run, we will need only the pre-trained multigrain model, that can be
downloaded from the github page or simply with

```
wget https://dl.fbaipublicfiles.com/multigrain/multigrain_models/joint_3B_0.5.pth -O data/multigrain_joint_3B_0.5.pth
```

There is no code dependency to the multigrain repository, we will just use
vanilla PyTorch code. Here again it is strongly advised to use a GPU.

## PCA+whitening training

On ouput of the resnet50, there is a 2048-dimensional descriptor.
Multigrain's accuracy depends strongly on a whitening transformation that makes
the descriptor's distribution in space more uniform.
The whitening transform is very similar to a PCA, except that each dimension
of the vector is normalized by its eigenvalue, hence the operation is also called PCA in the code.

To run the PCA training on 10000 vectors, do:

```bash
python baselines/GeM_baseline.py \
         --file_list list_files/train \
         --image_dir images/train \
         --pca_file data/pca_multigrain.vt \
         --n_train_pca 10000 \
         --train_pca

```

## Feature extraction

The feature extraction for the training set is very similar to that for GIST:

```bash
python baselines/GeM_baseline.py \
    --file_list list_files/subset_1_queries \
    --image_dir images/queries \
    --o data/subset_1_queries_multigrain.hdf5 \
    --pca_file data/pca_multigrain.vt

python baselines/GeM_baseline.py \
    --file_list list_files/subset_1_references \
    --image_dir images/references \
    --o data/subset_1_references_multigrain.hdf5 \
    --pca_file data/pca_multigrain.vt

```

And the descriptor-based evaluation is:
```
python scripts/compute_metrics.py \
    --query_descs data/subset_1_queries_multigrain.hdf5 \
    --db_descs data/subset_1_references_multigrain.hdf5 \
    --gt_filepath list_files/subset_1_ground_truth.csv \
    --track2 \
    --max_dim 2000
```

Giving:
```
Average Precision: 0.56314
Recall at P90    : 0.41374
Threshold at P90 : -1.51692
Recall at rank 1:  0.60970
Recall at rank 10: 0.70827
```

Which is significantly more accurate than GIST.

