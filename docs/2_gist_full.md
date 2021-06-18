
Tu use it, compile `compute_gist_stream` in the lear_gist-1.2 directory with the makefile. It depends on [fftw](http://www.fftw.org/) which is available on mainstream distributions.


## Running and evaluating

### GIST baseline for track #2

Here is the second version of the code package including:

GIST features are extracted in 3 stages:

1. PCA matrix computation

```bash
v2dir=/path/to/data
gist_exec=/path/to/compute_gist_stream

export PYTHONPATH=.

# pre-train PCA matrix

python baselines/gist_baseline.py \
     --nproc 20 \
     --giststream_exec $gist_exec \
     --pca_file  $v2dir/pca_baseline.vt \
     --image_dir $v2dir/1M_ref_images \
     --file_list $v2dir/references.csv \
     --train_pca

```

`--noproc` is to set the number of computation threads, adjust to your machine.


2. extract features for query images
```bash

python baselines/gist_baseline.py \
     --nproc 20 \
     --giststream_exec $gist_exec \
     --pca_file  $v2dir/pca_baseline.vt \
     --image_dir $v2dir/v2_all_queries_jpg \
     --file_list $v2dir/queries.csv \
     --o $v2dir/queries_gist_baseline.hdf5
```

3. extract features for reference images

```bash
python baselines/gist_baseline.py \
     --nproc 20 \
     --giststream_exec $gist_exec \
     --pca_file  $v2dir/pca_baseline.vt \
     --image_dir $v2dir/1M_ref_images \
     --file_list $v2dir/references.csv \
     --o $v2dir/references_gist_baseline.hdf5

```

4. evaluation

The descriptors can enter the evaluation script for track 2:

```bash
python scripts/compute_metrics.py  \
  --track2  \
  --query_descs $v2dir/queries_gist_baseline.hdf5 \
  --db_descs    $v2dir/references_gist_baseline.hdf5 \
  --query_list  $v2dir/queries.csv \
  --db_list     $v2dir/references.csv \
  --gt_filepath $v2_dir/gt.csv

```
And normally it should output
```
Track 2 running matching of 59916 queries in 1019936 database (256D descriptors), max_results=500000.
Evaluating 500000 predictions (19936 GT matches)
Average Precision: 0.19447
Recall at P90    : 0.15710
Threshold at P90 : -0.0643869
Recall at rank 1:  0.26124
Recall at rank 10: 0.26570
```


