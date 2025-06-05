"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import os
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {
            c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
        }
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def random_split_by_class(self, labels, train_per_class, test_per_class):
        """
        Randomly sample training and testing nodes according to assignment requirements
        """
        # Get unique classes
        unique_classes = np.unique(labels)

        # Group node indices by class
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_indices[label].append(idx)

        train_indices = []
        test_indices = []
        val_indices = []

        print(f"Dataset: {self.dataset_name}")
        print(f"Classes found: {unique_classes}")

        for class_label in unique_classes:
            indices = np.array(class_to_indices[class_label])
            print(f"Class {class_label}: {len(indices)} nodes")

            # Check if we have enough nodes for the required split
            required_nodes = train_per_class + test_per_class
            if len(indices) < required_nodes:
                print(
                    f"Warning: Class {class_label} has only {len(indices)} nodes, "
                    f"but need {required_nodes} (train: {train_per_class}, test: {test_per_class})"
                )

            # Randomly shuffle indices for this class
            np.random.shuffle(indices)

            # Sample training nodes
            train_indices.extend(indices[:train_per_class])

            # Sample testing nodes
            test_start = train_per_class
            test_end = train_per_class + test_per_class
            test_indices.extend(indices[test_start:test_end])

            # Use remaining nodes for validation (if any)
            if len(indices) > test_end:
                val_indices.extend(indices[test_end:])

        # Convert to numpy arrays and sort
        train_indices = np.sort(np.array(train_indices))
        test_indices = np.sort(np.array(test_indices))
        val_indices = np.sort(np.array(val_indices))

        print(
            f"Final splits - Train: {len(train_indices)}, Test: {len(test_indices)}, Val: {len(val_indices)}"
        )

        return train_indices, test_indices, val_indices

    # def load(self):
    #     """Load citation network dataset"""
    #     print("Loading {} dataset...".format(self.dataset_name))

    #     # load node data from file
    #     idx_features_labels = np.genfromtxt(
    #         "{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str)
    #     )
    #     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #     onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

    #     # load link data from file and build graph
    #     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    #     idx_map = {j: i for i, j in enumerate(idx)}
    #     reverse_idx_map = {i: j for i, j in enumerate(idx)}
    #     edges_unordered = np.genfromtxt(
    #         "{}/link".format(self.dataset_source_folder_path), dtype=np.int32
    #     )
    #     edges = np.array(
    #         list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    #     ).reshape(edges_unordered.shape)
    #     adj = sp.coo_matrix(
    #         (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #         shape=(onehot_labels.shape[0], onehot_labels.shape[0]),
    #         dtype=np.float32,
    #     )
    #     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #     norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

    #     # convert to pytorch tensors
    #     features = torch.FloatTensor(np.array(features.todense()))
    #     labels = torch.LongTensor(np.where(onehot_labels)[1])
    #     adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

    #     # the following part, you can either put them into the setting class or you can leave them in the dataset loader
    #     # the following train, test, val index are just examples, sample the train, test according to project requirements
    #     if self.dataset_name == "cora":
    #         idx_train = range(140)
    #         idx_test = range(200, 1200)
    #         idx_val = range(1200, 1500)
    #     elif self.dataset_name == "citeseer":
    #         idx_train = range(120)
    #         idx_test = range(200, 1200)
    #         idx_val = range(1200, 1500)
    #     elif self.dataset_name == "pubmed":
    #         idx_train = range(60)
    #         idx_test = range(6300, 7300)
    #         idx_val = range(6000, 6300)

    #     idx_train = torch.LongTensor(idx_train)
    #     idx_val = torch.LongTensor(idx_val)
    #     idx_test = torch.LongTensor(idx_test)
    #     # get the training nodes/testing nodes
    #     # train_x = features[idx_train]
    #     # val_x = features[idx_val]
    #     # test_x = features[idx_test]
    #     # print(train_x, val_x, test_x)

    #     train_test_val = {
    #         "idx_train": idx_train,
    #         "idx_test": idx_test,
    #         "idx_val": idx_val,
    #     }
    #     graph = {
    #         "node": idx_map,
    #         "edge": edges,
    #         "X": features,
    #         "y": labels,
    #         "utility": {"A": adj, "reverse_idx": reverse_idx_map},
    #     }
    #     return {"graph": graph, "train_test_val": train_test_val}
    def load(self):
        """Load citation network dataset"""
        print("Loading {} dataset...".format(self.dataset_name))

        # load node data from file
        try:
            idx_features_labels = np.genfromtxt(
                "{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str)
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find node file at {self.dataset_source_folder_path}/node"
            )

        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}

        try:
            edges_unordered = np.genfromtxt(
                "{}/link".format(self.dataset_source_folder_path), dtype=np.int32
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find link file at {self.dataset_source_folder_path}/link"
            )

        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
        ).reshape(edges_unordered.shape)
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(onehot_labels.shape[0], onehot_labels.shape[0]),
            dtype=np.float32,
        )
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # Get string labels for split calculation
        string_labels = idx_features_labels[:, -1]

        # Random sampling according to assignment requirements
        if self.dataset_name == "cora":
            # 7 classes, 20 train per class, 150 test per class
            train_per_class = 20
            test_per_class = 150
            expected_classes = 7
        elif self.dataset_name == "citeseer":
            # 6 classes, 20 train per class, 200 test per class
            train_per_class = 20
            test_per_class = 200
            expected_classes = 6
        elif self.dataset_name == "pubmed":
            # 3 classes, 20 train per class, 200 test per class
            train_per_class = 20
            test_per_class = 200
            expected_classes = 3
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Perform random split
        idx_train, idx_test, idx_val = self.random_split_by_class(
            string_labels, train_per_class, test_per_class
        )

        # Verify we have the expected number of nodes
        expected_train = train_per_class * expected_classes
        expected_test = test_per_class * expected_classes

        print(f"Expected train nodes: {expected_train}, Got: {len(idx_train)}")
        print(f"Expected test nodes: {expected_test}, Got: {len(idx_test)}")

        if len(idx_train) != expected_train:
            print(
                f"Warning: Expected {expected_train} training nodes, but got {len(idx_train)}"
            )
        if len(idx_test) != expected_test:
            print(
                f"Warning: Expected {expected_test} test nodes, but got {len(idx_test)}"
            )

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        train_test_val = {
            "idx_train": idx_train,
            "idx_test": idx_test,
            "idx_val": idx_val,
        }
        graph = {
            "node": idx_map,
            "edge": edges,
            "X": features,
            "y": labels,
            "utility": {"A": adj, "reverse_idx": reverse_idx_map},
        }
        return {"graph": graph, "train_test_val": train_test_val}

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)
