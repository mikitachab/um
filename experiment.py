import json

from numpy.lib.function_base import append
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import xgboost as xgb
import pandas as pd
import numpy as np


from tabulate import tabulate

from data import get_datasets_files

models = {
    "XGBClassifier": xgb.XGBClassifier(n_jobs=4, verbosity=0),
    "RandomForestClassifier": RandomForestClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "ExtraTreesClassifier": ExtraTreesClassifier(),
}



def main():
    print(get_datasets_files())
    print(models)
    r = run_exp(models, get_datasets_files())
    print_and_save_result("results", r, models)


def get_dataset_name(filename):
    return filename.split("/")[1].split(".")[0]


def run_exp(models, datasets):
    cv = RepeatedStratifiedKFold(random_state=1410, n_repeats=5, n_splits=2)
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
            print(name)
            scores = cross_val_score(
                model, X, y, scoring="f1_weighted", cv=cv, n_jobs=2
            )
            dataset_scores[name] = scores
    return final_scores


def print_and_save_result(result_name, results, models):
    def make_table(results, models):
        records = []
        for dataset_name, result in results.items():
            record = [dataset_name]
            for model in models:
                record.append(np.mean(result[model]))
            records.append(record)
        return tabulate(records, headers=["dataset"] + list(models.keys()))

    def save_result(table):
        with open(result_name + ".txt", "w") as file:
            file.write(table)

    def save_cv_results():
        data = {
            dataset: {model: list(cv_scores) for model, cv_scores in scores.items()}
            for dataset, scores in results.items()
        }
        with open(result_name + ".json", "w") as file:
            json.dump(data, file, indent=2)

    table = make_table(results, models)
    print(result_name)
    print(table)
    save_result(table)
    save_cv_results()


if __name__ == "__main__":
    main()
