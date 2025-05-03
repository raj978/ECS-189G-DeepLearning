"""
Concrete Evaluate class for a specific evaluation metrics
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from local_code.base_class.evaluate import evaluate


class Evaluate_Metrics(evaluate):
    data = None

    def eval_accuracy(self):
        y_true = self.data["true_y"]
        y_pred = self.data["pred_y"]

        # Compute various metrics
        accuracy = accuracy_score(y_true, y_pred)

        return accuracy

    def evaluate(self):
        print("evaluating performance...")
        y_true = self.data["true_y"]
        y_pred = self.data["pred_y"]

        # Compute various metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision_micro": precision_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_micro": recall_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        # Print metrics
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        return metrics

    def gen_learning_curve(self, hist, save_path):
        plt.plot(hist["epoch"], hist["accuracy"], label="Training accuracy")
        plt.plot(hist["epoch"], hist["loss"], label="Training loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
