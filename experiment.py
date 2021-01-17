from numpy.lib.function_base import append
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import xgboost as xgb
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import make_pipeline

from tabulate import tabulate

from data import get_datasets_files

baggin_vs_boosting_models = {
    # "XGBClassifier": xgb.XGBClassifier(n_jobs=1),
    "RandomForestClassifier": RandomForestClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
}

data_transoforms_models = {
    "RandomForestClassifier No Transforms": RandomForestClassifier(),
    "RandomForestClassifier + SMOTE": make_pipeline(
        SMOTE(k_neighbors=3), RandomForestClassifier()
    ),
    "RandomForestClassifier + ENN": make_pipeline(
        EditedNearestNeighbours(), RandomForestClassifier()
    ),
}


def main():
    r1 = run_exp(baggin_vs_boosting_models)
    print_result("exp 1", r1, baggin_vs_boosting_models)
    r2 = run_exp(data_transoforms_models)
    print_result("exp 2", r2, data_transoforms_models)


def get_dataset_name(filename):
    return filename.split("/")[1].split(".")[0]


def run_exp(models):
    cv = RepeatedStratifiedKFold(random_state=1410, n_repeats=5, n_splits=2)
    datasets = get_datasets_files()[5:7]
    print(datasets)
    final_scores = {}
    for dataset_path in datasets:
        print(dataset_path)
        ds_name = get_dataset_name(dataset_path)
        print(ds_name)
        df = pd.read_csv(dataset_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:].to_numpy().ravel()
        dataset_scores = final_scores.setdefault(ds_name, {})
        for name, model in models.items():
            scores = cross_val_score(model, X, y, scoring="f1_weighted", cv=cv)
            dataset_scores[name] = scores
    return final_scores


def print_result(result_name, results, models):
    records = []
    for dataset_name, result in results.items():
        record = [dataset_name]
        for model in models:
            record.append(np.mean(result[model]))
        records.append(record)
    table = tabulate(records, headers=["dataset"] + list(models.keys()))
    print(table)


if __name__ == "__main__":
    main()
