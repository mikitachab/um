# um

machine learning experiments

## Prepare env

```
source prepare_env.sh
```

## Data

The idea is to prepare all data needed for research by single command:
```
python data.py
```

This command should fetch and prepare/preprocess all datasets. Datasets should appear in `datasets` directory.

## Experiment

Run experiment

```
python experiment.py
```

`experiment.py` should run all experiments for all datasets and display results (statistical tests included).
