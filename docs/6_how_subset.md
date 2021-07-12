
# Extracting HOW local features on a small dataset

## HOW local features

Searching with HOW local features consists of two steps - HOW extraction and ASMK search. For this run, we use the method -R50<sub>how</sub> (n = 2000) from paper [Learning and aggregating deep local descriptors for instance-level recognition](https://arxiv.org/abs/2007.13172). To make use of the method, HOW package must be cloned and its dependencies installed, as described in the [HOW github repository](https://github.com/gtolias/how). All commands expect the how repository to be inside the `baselines` folder. It is necessary to use GPU for both extraction and search due to a limited performance on a CPU. Intermediate results are provided and can be downloaded for each step. Times reported in parentheses are for GPU Tesla V100.

```bash
cd baselines
git clone https://github.com/gtolias/how.git
# Follow instructions in https://github.com/gtolias/how
cd ..
export PYTHONPATH="baselines/asmk:baselines/cnnimageretrieval-pytorch-1.2:baselines/how:$PYTHONPATH"
```

## Training the codebook

Before the actual feature extraction, it is necessary to train the codebook *(4 min)*:

```bash
python baselines/how/examples/demo_how.py \
    eval \
    baselines/how_r50-_2000.yml \
    --step train_codebook
```

Alternatively, the codebook can be downloaded:

```bash
mkdir -p "data/how_r50-_2000/eval"
wget "http://ptak.felk.cvut.cz/personal/jenicto2/download/isc2021_how_r50-_2000/eval/codebook.pkl" -P "data/how_r50-_2000/eval/"
```

## Building the database

Each image is described by a set of 512-dimensional binary descriptors. To extract the features for the reference images, run *(9 min)*:

```bash
python baselines/how/examples/demo_how.py \
    eval \
    baselines/how_r50-_2000.yml \
    --datasets-local '[{"name": "subset_1_references", "image_root": "../images/references/*.jpg", "query_list": null, "database_list": "list_files/subset_1_references"}]' \
    --step aggregate_database
```

The extracted descriptors are used to build the inverted file. This is performed by changing the `--step` argument *(1 min)*:

```bash
python baselines/how/examples/demo_how.py \
    eval \
    baselines/how_r50-_2000.yml \
    --datasets-local '[{"name": "subset_1_references", "image_root": "../images/references/*.jpg", "query_list": null, "database_list": "list_files/subset_1_references"}]' \
    --step build_ivf
```

Alternatively, the inverted file can be downloaded:

```bash
wget "http://ptak.felk.cvut.cz/personal/jenicto2/download/isc2021_how_r50-_2000/eval/subset_1_references.ivf.pkl" -P "data/how_r50-_2000/eval/"
```

## Searching the queries

Query results can be searched in the inverted file *(25 min)*:

```bash
python baselines/how/examples/demo_how.py \
    eval \
    baselines/how_r50-_2000.yml \
    --datasets-local '[{"name": "subset_1_references", "image_root": "../images/queries/*.jpg", "query_list": "list_files/subset_1_queries", "database_list": null}]' \
    --step query_ivf
```

Alternatively, the query results can be downloaded:

```bash
wget "http://ptak.felk.cvut.cz/personal/jenicto2/download/isc2021_how_r50-_2000/eval/subset_1_references.results.pkl" -P "data/how_r50-_2000/eval/"
```

## Evaluation

Top matches can be identified in the search results:

```bash
python baselines/how_find_matches.py \
    baselines/how_r50-_2000.yml \
    subset_1_references \
    --query_list list_files/subset_1_queries \
    --db_list list_files/subset_1_references \
    --preds_filepath data/predictions_how_subset_1.csv
```

The matches are evaluated with:

```bash
python scripts/compute_metrics.py \
    --gt_filepath list_files/subset_1_ground_truth.csv \
    --preds_filepath data/predictions_how_subset_1.csv
```

Giving:
```
Average Precision: 0.49937
Recall at P90    : 0.35925
Threshold at P90 : 0.0105103
Recall at rank 1:  0.64436
Recall at rank 10: 0.73492
```

Which is more accurate than GIST, but less than MultiGrain.

