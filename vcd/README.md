# Installation (without VCSL)

```
conda create --name vcd -c pytorch -c conda-forge pytorch torchvision \
  scikit-learn numpy pandas matplotlib faiss
```

We don't need pytorch for the codebase currently; this is just the environment I used.

# Running tests

```
$ cd vcd
$ python -m unittest discover
..ss.................
----------------------------------------------------------------------
Ran 21 tests in 0.060s

OK (skipped=2)
```

# Descriptor eval

```
$ python -m vcd.descriptor_eval --query_path ../vcd_eval_data/queries.npz --ref_path ../vcd_eval_data/refs.npz --gt_path ../vcd_eval_data/gt.csv
Starting Descriptor level eval
Loaded 997 query features
Loaded 4974 ref features
Performing KNN with k: 5
Loaded GT from ../vcd_eval_data/gt.csv
Micro AP : 0.79020867778352
```

# Matching track eval

```
$ python -c 'from vcd.metrics import evaluate_matching_track; metrics = evaluate_matching_track("../vcd_eval_data/gt.csv", "../vcd_eval_data/matches.csv"); print(f"matching ap = {metrics.segment_ap_v2.ap}")'
matching ap = 0.5047952268737359
```
