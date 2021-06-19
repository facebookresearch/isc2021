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
pairs with the matching score of the same query with an image that is known to be
negative.
If the score is significantly better it is more likely that the match is true match.
The new score is then

score = old_score - factor * negative_image_score

One way to obtain scores for negative queries is to match the
query with the set of training images.
To get a more stable estimate of the negative score, it's also better to
take the score at rank 10 (or similar, not 1) of the training set.
Note that other reference images cannot be used to estimate the negative score
because the scoring of images should be independent.

### Extracting features on the training set


Computing the descriptors of 1~million training images is slow but straightforward:

```bash

for i in {0..19}; do
     python baselines/GeM_baseline.py \
          --file_list list_files/train \
          --i0 $((i * 50000)) --i1 $(((i + 1) * 50000)) \
          --image_dir images/train \
          --o data/train_${i}_multigrain.hdf5 \
          --pca_file data/pca_multigrain.vt
done

```

### Subtracting scores

Then the score normalization can be performed with:

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
    --gt_filepath list_files/public_ground_truth.csv
```

Which gives
```
Track 1 results of 500000 predictions (4991 GT matches)
Average Precision: 0.36491
Recall at P90    : 0.26828
Threshold at P90 : -0.014896
Recall at rank 1:  0.44961
Recall at rank 10: 0.49309
```

So this basic score normalization results a factor 2 increase of the
average precision.
This shows that score normalization is important for this kind of descriptors.
Incorporating a distance calibration at training time should make it possible to
obtain better results without this kind of matching-time patches.

<!--

python scripts/score_normalization.py \
    --query_descs data/dev_queries_multigrain.hdf5 \
    --db_descs data/references_{0..19}_multigrain.hdf5 \
    --train_descs data/train_{0..19}_multigrain.hdf5 \
    --factor 2.0 --n 10 \
    --o data/predictions_dev_queries_normalized.csv


python scripts/compute_metrics.py \
    --preds_filepath data/predictions_dev_queries_normalized.csv \
    --gt_filepath list_files/full_ground_truth.csv

Average Precision: 0.36420
Recall at P90    : 0.27200
Threshold at P90 : -0.015121
Recall at rank 1:  0.44520
Recall at rank 10: 0.48300

-->



