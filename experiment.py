import json
import warnings

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
from scipy import stats
from tabulate import tabulate

from data import get_datasets_files

warnings.filterwarnings("ignore")


models = {
    "XGBClassifier": xgb.XGBClassifier(n_jobs=4, verbosity=0, silent=True),
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


def make_table(results, models, t_tests):
    records = []
    for dataset_name, result in results.items():
        dataset_t_tests = t_tests[dataset_name]
        record = [dataset_name]
        for model in models:
            val = str(round(np.mean(result[model]), 3))
            t = ",".join(str(x) for x in dataset_t_tests[model])
            record.append(f"{val}\n{t}")
        records.append(record)
    return tabulate(
        records,
        headers=["dataset"] + list(models.keys()),
        numalign="center",
        stralign="center",
        tablefmt="grid",
    )


def save_result(name, table):
    with open(name + ".txt", "w") as file:
        file.write(table)


def save_cv_results(name, results):
    data = {
        dataset: {model: list(cv_scores) for model, cv_scores in scores.items()}
        for dataset, scores in results.items()
    }
    with open(name + ".json", "w") as file:
        json.dump(data, file, indent=2)


def print_and_save_result(result_name, results, models):
    datasets = list(results.keys())
    t_tests = make_t_tests(results, models, datasets)
    table = make_table(results, models, t_tests)
    print(result_name)
    print(table)
    save_result(result_name, table)
    save_cv_results(result_name, results)


def make_t_tests(cv_results, models, datasets):
    results = {}
    for dataset in datasets:
        dataset_data = cv_results[dataset]
        results[dataset] = {}
        for model_a in models:
            results[dataset][model_a] = []
            for i, model_b in enumerate(models, 1):
                if model_a != model_b:
                    t, a = stats.ttest_ind(dataset_data[model_a], dataset_data[model_b])
                    if a <= 0.05 and t > 0:
                        results[dataset][model_a].append(i)
    return results


if __name__ == "__main__":
    main()
