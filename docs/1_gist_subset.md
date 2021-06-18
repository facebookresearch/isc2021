# GIST descriptors on a small subset

## Subset 1

The number of queries in the development set is 50k, and we have the ground truth for 25k of
these queries (in the file list_files/public_ground_truth.csv).
The number of reference images to search in is 1 million.

For a quick test, we are going to use much fewer images than that: only the images from pairs of known matches.
This excludes all distractor images, both on the query and reference side, which makes the problem much easier.

Note: To run this test, you do not need to have the data downloaded, as the subset for the test is in this repo in `list_files/`.

The subset is defined in three files:

- `list_files/subset_1_queries`: list of query ids. The ids are just of the form Q00123, the filename without .jpg extension.

- `list_files/subset_1_references`: list of reference ids

- `list_files/subset_1_ground_truth.csv`: ground-truth matches between queries and references

## running feature extraction

The script `baselines/gist_baseline.py` performs GIST feature extraction.
You can call it on the query images via:
```bash
python baselines/gist_baseline.py \
    --file_list list_files/subset_1_queries \
    --image_dir images/queries \
    --o data/subset_1_queries_gist.hdf5 \
    --nproc 20
```
the `--nproc 20` should be adjusted to the number of cores on your machine.
The output is in hdf5 format.

Similarly, for the reference images, run:
```bash
python baselines/gist_baseline.py \
    --file_list list_files/subset_1_references \
    --image_dir images/references \
    --o data/subset_1_references_gist.hdf5 \
    --nproc 20
```

Note that the feature extraction script outputs the following information:
```bash
hardware_image_description: model name : Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, 80 cores
image_description_time: 0.00088 s per image
```
This information can be copy/pasted to the `descriptor-track-metadata.txt` metadata file.

## Evaluation

The `scripts/compute_metrics.py` can evaluate these results as if they were a track 2 submission:
```bash
python scripts/compute_metrics.py \
    --query_descs data/subset_1_queries_gist.hdf5 \
    --db_descs data/subset_1_references_gist.hdf5 \
    --gt_filepath list_files/subset_1_ground_truth.csv \
    --track2 \
    --max_dim 1000
```
Note that this is not a valid track 2 submission because the dimension of the vectors is
960, larger than the allowed 256 (hence the `--max_dim` argument to override this check).

The output looks like:
```bash
Track 2 running matching of 4991 queries in 4991 database (960D descriptors), max_results=500000.
Evaluating 500000 predictions (4991 GT matches)
Average Precision: 0.27063
Recall at P90    : 0.22501
Threshold at P90 : -0.149336
Recall at rank 1:  0.33380
Recall at rank 10: 0.37808
```

The threshold is a negated L2 distance (because scores should be higher for low L2 distance).

By adding option `--write_predictions data/predictions_subset_1_gist.csv`, to the script call above, you get a valid track 1 submission:
```bash
python scripts/compute_metrics.py \
    --preds_filepath data/predictions_subset_1_gist.csv \
    --gt_filepath list_files/subset_1_ground_truth.csv
```
which give exactly the same output as the track 2 run above.

This is also a good way to look at the results (they can be sorted on the command line with `sort -n -r -t , -k 3 data/predictions_subset_1_gist.csv`).
For example, one of the top results is:
```bash
Q10893,R751771,-0.008623
```
Which corresponds to the following images:

![images/queries/Q10893.jpg](img/Q10893.jpg)
![images/references/Q10893.jpg](img/R751771.jpg)

## Evaluation with the official script -- track 1

There is an official evaluation script available on the Driven Data site here:
https://www.drivendata.org/competitions/80/competition-image-similarity-2-dev/data/

For track 1, the official evaluation script can consume the predictions_subset_1.csv directly:
```bash
python eval_metrics.py \
        data/predictions_subset_1_gist.csv \
        list_files/subset_1_ground_truth.csv
```
Which outputs the same results:
```bash
{
  "average_precision": 0.27063061936539407,
  "recall_p90": 0.22500500901622922
}
```

## Evaluation with the official script -- track 2

The submission format for track 2 is different from the descriptor format we produced.
There is a small tool to convert separate two separate hdf5 files to the official format with a single file.

Run the conversion with:
```bash
python scripts/convert_track2_format.py \
    --query_descs data/subset_1_queries_gist.hdf5 \
    --db_descs data/subset_1_references_gist.hdf5 \
    --o data/subset_1_descriptors.hdf5
```

Then evaluate with:
```bash
python eval_metrics.py \
        data/subset_1_descriptors.hdf5 \
        list_files/subset_1_ground_truth.csv
```
Which outputs:
```bash
{
  "average_precision": 0.2694426487535491,
  "recall_p90": 0.22500500901622922
}
```
(the reason of this slight descrepancy is being investigated...)