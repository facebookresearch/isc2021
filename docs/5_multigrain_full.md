# MultiGrain on the full dataset

## Extracting features

Also on 25k queries

```
python baselines/GeM_baseline.py \
    --file_list list_files/dev_queries_25k \
    --image_dir images/queries \
    --o data/dev_queries_25k_multigrain.hdf5 \
    --pca_file data/pca_multigrain.vt

```

20 minutes per 50k images

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
This is a symptom of image distances that are not comparable.


## Score normalization

We need to apply a score normalization so that the scores for different query
images are comparable.
The difficulty here is that the

### Extracting features on the training set


### subtracting scores