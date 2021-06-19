# MultiGrain on the full dataset


## Extracting features


Feature extraction is very similar to GIST.
The difference is that the descriptors are much slower to extract
(about 20 minutes of computation for 50k images on a state-of-the-art GPU).

Here it is really useful to break down the computation into processing batches
for the 1 million reference images.

```bash

# extract for query images
python baselines/GeM_baseline.py \
    --file_list list_files/dev_queries \
    --image_dir images/queries \
    --o data/dev_queries_multigrain.hdf5 \
    --pca_file data/pca_multigrain.vt

# reference images
for i in {0..19}; do
     python baselines/GeM_baseline.py \
          --file_list list_files/references \
          --i0 $((i * 50000)) --i1 $(((i + 1) * 50000)) \
          --image_dir images/references \
          --o data/references_${i}_multigrain.hdf5 \
          --pca_file data/pca_multigrain.vt
done

```

Also on 25k queries:

```bash
python baselines/GeM_baseline.py \
    --file_list list_files/dev_queries_25k \
    --image_dir images/queries \
    --o data/dev_queries_25k_multigrain.hdf5 \
    --pca_file data/pca_multigrain.vt

```

## Evaluation

We can run the evaluation on the 25k subset.

```bash
python scripts/compute_metrics.py \
    --query_descs data/dev_queries_25k_multigrain.hdf5 \
    --db_descs data/references_{0..19}_multigrain.hdf5 \
    --gt_filepath list_files/public_ground_truth.csv \
    --max_dim 2000 \
    --track2
```

The ouptut is

```
Track 2 running matching of 25000 queries in 1000000 database (1500D descriptors), max_results=500000.
Evaluating 500000 predictions (4991 GT matches)
Average Precision: 0.15402
Recall at P90    : 0.03426
Threshold at P90 : -0.520392
Recall at rank 1:  0.44861
Recall at rank 10: 0.48387
```

A bit disappointing: not much better than GIST!
However, the recall at rank 1 and 10, the performance *is* much better.
Recalls are per-query statistics and they are computed only on images that have
actual matches.
This is a symptom of image distances that are not comparable.


## Score normalization

We need to apply a score normalization so that the scores for different query
images are comparable.
For this, a basic technique is to compare the scores of (query, reference) image
pairs with the matching score of the query with an image that is known to be
negative.

One way to do this is to


```bash
python scripts/score_normalization.py \
    --query_descs data/dev_queries_25k_multigrain.hdf5 \
    --db_descs data/references_{0..19}_multigrain.hdf5 \
    --train_descs data/train_{0..19}_multigrain.hdf5 \
    --factor 2.0 --n 10 \
    --o data/predictions_dev_queries_25k_normalized.csv
```

Then evaluate with

```bash
python scripts/compute_metrics.py \
    --preds_filepath data/predictions_dev_queries_25k_normalized.csv \
    --gt_filepath list_files/public_ground_truth.csv \
```



### Extracting features on the training set


### subtracting scores