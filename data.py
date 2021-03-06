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
]

class CreditCardFraudDetectionKaggleDataset:
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


class PortoSegurosSafeDriverPredictionKaggleDataset:
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
    preprocessing_1()
    preprocessing_2()


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

def preprocessing_1():
    df = pd.read_csv('datasets/ecoli.csv', header=None)
    df.rename(columns={0: 'id0', 1: 'id1', 2: 'id2', 3: 'id3', 4: 'id4', 5: 'id5', 6: 'id6', 7: 'id7',  }, inplace=True)
    df.to_csv('datasets/ecoli.csv', index=False)
    le = preprocessing.LabelEncoder()
    df['id7'] = le.fit_transform(df['id7'])
    df.to_csv('datasets/ecoli.csv', index=False)

def preprocessing_2():
    df2 = pd.read_csv('datasets/german.csv', header=None)
    df2 = df2.astype(str)
    df2 = df2.replace('A', '', regex=True)
    df2.to_csv('datasets/german.csv', index=False)

if __name__ == "__main__":
    main()
