from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import xgboost as xgb
import pandas as pd

from data import get_datasets_files

models = [
    # xgb.XGBClassifier(n_jobs=1),
    RandomForestClassifier(),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(),
]


def main():
    """
    Experiment 1
    """
    cv = RepeatedStratifiedKFold(random_state=1410, n_repeats=5, n_splits=2)
    datasets = get_datasets_files()
    print(datasets)
    final_scores = {}
    for dataset_path in datasets:
        print(dataset_path)
        df = pd.read_csv(dataset_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:].to_numpy().ravel()
        for model in models:
            scores = cross_val_score(model, X, y, scoring="f1_weighted")
            print(scores)


if __name__ == "__main__":
    main()
