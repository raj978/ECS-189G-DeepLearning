"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_train_file_path = ""
    dataset_test_file_path = ""

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self, train=True):
        fname = self.dataset_train_file_path if train else self.dataset_test_file_path

        X, y = [], []
        with open(fname, "r") as f:
            for line in f:
                row = line.strip().split(",")
                y.append(int(row[0]))
                X.append([int(v) for v in row[1:]])
        return {"X": X, "y": y}
