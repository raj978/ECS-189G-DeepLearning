"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import json

import numpy as np
import torch
from torch import nn

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    model_path = ""
    hist_path = ""

    architecture = "small"
    loss_fn = "cross"
    opt = "adam"

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    # def __init__(self, mName, mDescription):
    #     method.__init__(self, mName, mDescription)
    #     nn.Module.__init__(self)
    #     # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    #     self.fc_layer_1 = nn.Linear(784, 128)
    #     # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    #     self.activation_func_1 = nn.ReLU()
    #     self.fc_layer_2 = nn.Linear(128, 10)
    #     # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    #     self.activation_func_2 = nn.Softmax(dim=1)
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # choose one of several preset architectures
        self.net = self.build_model(self.architecture)

        losses = {
            "cross": nn.CrossEntropyLoss(),
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
        }
        self.loss_function = losses[self.loss_fn]
        self.opt = self.opt

    def build_model(self, arch):
        layers = []
        if arch == "small":
            dims = [784, 128, 10]
        elif arch == "medium":
            dims = [784, 256, 128, 64, 10]
        elif arch == "wide":
            dims = [784, 512, 512, 256, 10]
        else:
            raise ValueError(f"Unknown arch: {arch}")

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # add activation on all but last layer
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.LeakyReLU(0.1, inplace=True))
                layers.append(nn.Dropout(0.5))
        # for classification add log-softmax
        layers.append(nn.LogSoftmax(dim=1))
        return nn.Sequential(*layers)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    # def forward(self, x):
    #     """Forward propagation"""
    #     # hidden layer embeddings
    #     h = self.activation_func_1(self.fc_layer_1(x))
    #     # outout layer result
    #     # self.fc_layer_2(h) will be a nx2 tensor
    #     # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
    #     # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
    #     y_pred = self.activation_func_2(self.fc_layer_2(h))
    #     return y_pred
    def forward(self, x):
        return self.net(x)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        if self.opt == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.opt == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9
            )

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Metrics("training evaluator", "")

        history = {"epoch": [], "loss": [], "accuracy": []}

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(
            self.max_epoch
        ):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = self.loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            accuracy_evaluator.data = {"true_y": y_true, "pred_y": y_pred.max(1)[1]}
            acc = accuracy_evaluator.eval_accuracy()
            loss = train_loss.item()

            history["epoch"].append(epoch)
            history["loss"].append(loss)
            history["accuracy"].append(acc)

            if epoch % 100 == 0:
                print(
                    "Epoch:",
                    epoch,
                    "Accuracy:",
                    acc,
                    "Loss:",
                    loss,
                )
            else:
                print(
                    f"Epoch {epoch+1}/{self.max_epoch}  loss: {loss:.4f}  acc: {acc:.4f}",
                    end="\r",
                    flush=True,
                )

        return history

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print("method running...")
        print("--start training...")
        history = self.train(self.data["train"]["X"], self.data["train"]["y"])
        print("\n--start testing...")
        pred_y = self.test(self.data["test"]["X"])

        torch.save(self, self.model_path)
        with open(self.hist_path, "w") as f:
            json.dump(history, f)

        return history, {"pred_y": pred_y, "true_y": self.data["test"]["y"]}
