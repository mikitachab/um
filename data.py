import os
from os.path import join
import subprocess
from dataclasses import dataclass
from sklearn import preprocessing
import csv
import re

import pandas as pd

DATASETS_DIR = "datasets"


@dataclass
class SimpleCsvDatasetInfo:
    name: str
    target_filename: str
    csv_url: str


simple_datasets_info = [
    SimpleCsvDatasetInfo(
        name="Pima Indians Diabetes",
        target_filename="pima-indians-diabetes.csv",
        csv_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
    ),
    SimpleCsvDatasetInfo(
        name="Haberman Breast Cancer",
        target_filename="haberman.csv",
        csv_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv",
    ),
    SimpleCsvDatasetInfo(
        name="German Credit",
        target_filename="german.csv",
        csv_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv",
    ),
    SimpleCsvDatasetInfo(
        name="Glass Identification",
        target_filename="glass.csv",
        csv_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv",
    ),
    SimpleCsvDatasetInfo(
        name="E-coli",
        target_filename="ecoli.csv",
        csv_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/ecoli.csv",
    ),
    SimpleCsvDatasetInfo(
        name="Thyroid Gland",
        target_filename="new-thyroid.csv",
        csv_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/new-thyroid.csv",
    ),
    SimpleCsvDatasetInfo(
        name="Iris",
        target_filename="Iris.csv",
        csv_url="https://raw.githubusercontent.com/Student235555/some-datasets/master/datasets/Iris.csv",
    ),
    SimpleCsvDatasetInfo(
        name="Abalone",
        target_filename="abalone.csv",
        csv_url="https://raw.githubusercontent.com/Student235555/some-datasets/master/datasets/abalone.csv",
    ),
    SimpleCsvDatasetInfo(
        name="banknote",
        target_filename="banknote.csv",
        csv_url="https://raw.githubusercontent.com/Student235555/some-datasets/master/datasets/banknote.csv",
    ),
    SimpleCsvDatasetInfo(
        name="breast-cancer-wisconsin",
        target_filename="breast-cancer-wisconsin.csv",
        csv_url="https://raw.githubusercontent.com/Student235555/some-datasets/master/datasets/breast-cancer-wisconsin.csv",
    ),
    SimpleCsvDatasetInfo(
        name="cleveland",
        target_filename="cleveland.csv",
        csv_url="https://raw.githubusercontent.com/Student235555/some-datasets/master/datasets/cleveland.csv",
    ),
    SimpleCsvDatasetInfo(
        name="heart",
        target_filename="heart.csv",
        csv_url="https://raw.githubusercontent.com/Student235555/some-datasets/master/datasets/heart.csv",
    ),
    SimpleCsvDatasetInfo(
        name="sonar",
        target_filename="sonar.csv",
        csv_url="https://raw.githubusercontent.com/Student235555/some-datasets/master/datasets/sonar.csv",
    ),
    SimpleCsvDatasetInfo(
        name="wine",
        target_filename="wine.csv",
        csv_url="https://raw.githubusercontent.com/Student235555/some-datasets/master/datasets/wine.csv",
    ),
]


class DatasetInterface:
    @property
    def name(self) -> str:
        """
        should return name of dataset
        """
        raise NotImplementedError

    def prepare(self, dirname: str):
        """
        should fetch and prepare data
        Args:
            dirname (str): dir where dataset csv should be saved
        """
        raise NotImplementedError

class CreditCardFraudDetectionKaggleDataset(DatasetInterface):

    @property
    def name(self):
        return "Credit Card Fraud Detection"

    def prepare(self, dirname):
        kaggle_cmd = ["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud"]
        subprocess.run(kaggle_cmd, check=True)
        subprocess.run(["unzip", "creditcardfraud.zip"])
        filename = os.path.join(dirname, "creditcard.csv")
        subprocess.run(["mv", "creditcard.csv", filename])
        subprocess.run(["rm", "creditcardfraud.zip"])


class PortoSegurosSafeDriverPredictionKaggleDataset(DatasetInterface):
    @property
    def name(self):
        return "Porto Seguro’s Safe Driver Prediction"

    def prepare(self, dirname):
        kaggle_cmd = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "porto-seguro-safe-driver-prediction",
        ]
        subprocess.run(kaggle_cmd, check=True)
        subprocess.run(["unzip", "porto-seguro-safe-driver-prediction.zip"])
        out_df = self.join_train_and_test()
        filename = os.path.join(dirname, "porto-seguro-safe-driver-prediction.csv")
        out_df.to_csv(filename)

        self.cleanup()

    def join_train_and_test(self):
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        return train_df.append(test_df)

    def cleanup(self):
        subprocess.run(["rm", "porto-seguro-safe-driver-prediction.zip"])
        subprocess.run(["rm", "train.csv"])
        subprocess.run(["rm", "test.csv"])
        subprocess.run(["rm", "sample_submission.csv"])


kaggle_datasets = [
    CreditCardFraudDetectionKaggleDataset(),
    PortoSegurosSafeDriverPredictionKaggleDataset(),
]


def prepare_datasets(dirname=DATASETS_DIR):
    preare_simple_datasets(simple_datasets_info, dirname)
    prepare_kaggle_datasets(kaggle_datasets, dirname)
    preprocessing_ecoli()
    preprocessing_german()


def preare_simple_datasets(datasets_info, dirname):
    os.makedirs(dirname, exist_ok=True)

    for dataset_info in datasets_info:
        print(f"downloading {dataset_info.name}")
        filename = os.path.join(dirname, dataset_info.target_filename)
        wget_cmd = ["wget", f"--output-document={filename}", dataset_info.csv_url]

        subprocess.run(wget_cmd, check=True)


def prepare_kaggle_datasets(kaggle_datasets, dirname):
    for dataset in kaggle_datasets:
        dataset.prepare(dirname)


def main():
    prepare_datasets()


def get_datasets_files():
    return [os.path.join(DATASETS_DIR, file) for file in os.listdir(DATASETS_DIR)]


def preprocessing_ecoli():
    df = pd.read_csv("datasets/ecoli.csv", header=None)
    df.rename(
        columns={
            0: "id0",
            1: "id1",
            2: "id2",
            3: "id3",
            4: "id4",
            5: "id5",
            6: "id6",
            7: "id7",
        },
        inplace=True,
    )
    df.to_csv("datasets/ecoli.csv", index=False)
    le = preprocessing.LabelEncoder()
    df["id7"] = le.fit_transform(df["id7"])
    df.to_csv("datasets/ecoli.csv", index=False)


def preprocessing_german():
    def to_numbers():
        df2 = pd.read_csv("datasets/german.csv", header=None)
        df2 = df2.astype(str)
        df2 = df2.replace("A", "", regex=True)
        df2.to_csv("datasets/german.csv", index=False)

    def fix_classes():
        df2 = pd.read_csv("datasets/german.csv")
        df2["20"] = df2["20"].apply(lambda x: x + 1)
        df2.to_csv("datasets/german.csv", index=False)

    to_numbers()
    fix_classes()


def preprocessing_porto():
    df = pd.read_csv("datasets/porto-seguro-safe-driver-prediction.csv")
    df.dropna(inplace=True)
    df.to_csv("datasets/porto-seguro-safe-driver-prediction.csv", index=False)


if __name__ == "__main__":
    preprocessing_german()
