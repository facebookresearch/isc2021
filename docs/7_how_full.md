# How local features on the full dataset

The steps are identical as when processing only a subset in [Step 6](6_how_subset.md). For the most demanding steps, data can be processed in parallel, which is beneficial when more GPUs are available. Intermediate results are provided and can be downloaded for each step. Times reported in parentheses are for GPU Tesla V100.

If not performed for Step 6, the HOW package must be cloned and its dependencies installed:

```bash
cd baselines
git clone https://github.com/gtolias/how.git
# Follow instructions in https://github.com/gtolias/how
cd ..
export PYTHONPATH="baselines/asmk:baselines/cnnimageretrieval-pytorch-1.2:baselines/how:$PYTHONPATH"
```

## Training the codebook

If not performed for Step 6, the codebook must be trained *(4 min)*:

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

For extraction, the lists of database images can be partitioned and the partitions processed in parallel, independently of each other *(100 x 18 min)*:

```bash
PARTITIONS=100
for i in {0..$PARTITIONS}; do
    python baselines/how/examples/demo_how.py \
        eval \
        baselines/how_r50-_2000.yml \
        --datasets-local '[{"name": "references", "image_root": "../images/references/*.jpg", "query_list": null, "database_list": "list_files/references"}]' \
        --step aggregate_database \
        --partition ${PARTITIONS}_$i
done
```

After descriptors are extracted for all partitions, the inverted file can be built *(127 min)*:

```bash
python baselines/how/examples/demo_how.py \
    eval \
    baselines/how_r50-_2000.yml \
    --datasets-local '[{"name": "references", "image_root": "../images/references/*.jpg", "query_list": null, "database_list": "list_files/references"}]' \
    --step build_ivf
```

Alternatively, the inverted file can be downloaded:

```bash
wget "http://ptak.felk.cvut.cz/personal/jenicto2/download/isc2021_how_r50-_2000/eval/references.ivf.pkl" -P "data/how_r50-_2000/eval/"
```

## Searching the queries

Searching for query results can be also partitioned *(100 x 62 min)*:

```bash
PARTITIONS=100
for i in {0..$PARTITIONS}; do
    python baselines/how/examples/demo_how.py \
        eval \
        baselines/how_r50-_2000.yml \
        --datasets-local '[{"name": "references", "image_root": "../images/queries/*.jpg", "query_list": "list_files/dev_queries", "database_list": null}]' \
        --step query_ivf \
        --partition ${PARTITIONS}_$i
done
```

Alternatively, the search results can be downloaded:

```bash
wget "http://ptak.felk.cvut.cz/personal/jenicto2/download/isc2021_how_r50-_2000/eval/references.results.pkl" -P "data/how_r50-_2000/eval/"
```

## Evaluation

Top matches can be identified in the search results:

```bash
python baselines/how_find_matches.py \
    baselines/how_r50-_2000.yml \
    references \
    --query_list list_files/dev_queries \
    --db_list list_files/references \
    --preds_filepath data/predictions_how_nonorm.csv
```

Resulting:

```
Track 1 results of 500000 predictions (10000 GT matches)
Average Precision: 0.17327
Recall at P90    : 0.10540
Threshold at P90 : 0.0581182
Recall at rank 1:  0.36650
Recall at rank 10: 0.37440
```

This is only slightly better than GIST.

## Train set for score normalization

As with MultiGrain, HOW greatly benefits from score normalization. First, the descriptors for the `train` set needs to be extracted *(100 x 18 min)*:

```bash
PARTITIONS=100
for i in {0..$PARTITIONS}; do
    python baselines/how/examples/demo_how.py \
        eval \
        baselines/how_r50-_2000.yml \
        --datasets-local '[{"name": "train", "image_root": "../images/train/*.jpg", "query_list": null, "database_list": "list_files/train"}]' \
        --step aggregate_database \
        --partition ${PARTITIONS}_$i
done
```

Then, the corresponding inverted file can be built *(124 min)*:

```bash
python baselines/how/examples/demo_how.py \
    eval \
    baselines/how_r50-_2000.yml \
    --datasets-local '[{"name": "train", "image_root": "../images/train/*.jpg", "query_list": null, "database_list": "list_files/train"}]' \
    --step build_ivf
```

And finally, queries can be searched *(100 x 62 min)*:

```bash
PARTITIONS=100
for i in {0..$PARTITIONS}; do
    python baselines/how/examples/demo_how.py \
        eval \
        baselines/how_r50-_2000.yml \
        --datasets-local '[{"name": "train", "image_root": "../images/queries/*.jpg", "query_list": "list_files/dev_queries", "database_list": null}]' \
        --step query_ivf \
        --partition ${PARTITIONS}_$i
done
```

Alternatively, the inverted file and/or the search results can be downloaded:

```bash
wget "http://ptak.felk.cvut.cz/personal/jenicto2/download/isc2021_how_r50-_2000/eval/train.ivf.pkl" -P "data/how_r50-_2000/eval/"
wget "http://ptak.felk.cvut.cz/personal/jenicto2/download/isc2021_how_r50-_2000/eval/train.results.pkl" -P "data/how_r50-_2000/eval/"
```

## Evaluation of score normalization

The `references` search results can be normalized by the `train` set results:

```bash
python baselines/how_find_matches.py \
    baselines/how_r50-_2000.yml \
    references \
    --normalization_set train \
    --query_list list_files/dev_queries \
    --db_list list_files/references \
    --preds_filepath data/predictions_how.csv
```

Giving the final results:
```
Track 1 results of 500000 predictions (10000 GT matches)
Average Precision: 0.37156
Recall at P90    : 0.20350
Threshold at P90 : 0.0176242
Recall at rank 1:  0.47850
Recall at rank 10: 0.49310
```

The performance gain from normalization is substantial. This validates the necessity of utilizing some form of score normalization.

