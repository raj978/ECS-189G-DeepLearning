from __future__ import division, print_function

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument(
    "--fastmode",
    action="store_true",
    default=False,
    help="Validate during training pass.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--hidden", type=int, default=16, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--dataset", type=str, default="", help="Choose between citeseer, cora, and pubmed"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not args.dataset:
    raise Exception("No dataset selected")

loader = Dataset_Loader(dName=args.dataset)
loader.dataset_source_folder_path = f"../../data/stage_5_data/{args.dataset}"

# Load data
loaded = loader.load()
adj = loaded["graph"]["utility"]["A"]
features = loaded["graph"]["X"]
labels = loaded["graph"]["y"]
idx_train = loaded["train_test_val"]["idx_train"]
idx_val = loaded["train_test_val"]["idx_val"]
idx_test = loaded["train_test_val"]["idx_test"]

# Model and optimizer
model = GCN(
    nfeat=features.shape[1],
    nhid=args.hidden,
    nclass=labels.max().item() + 1,
    dropout=args.dropout,
)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# Initialize lists to store training history BEFORE the training loop
train_losses, val_losses = [], []
train_accs, val_accs = [], []


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = loader.accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = loader.accuracy(output[idx_val], labels[idx_val])

    print(
        "Epoch: {:04d}".format(epoch + 1),
        "loss_train: {:.4f}".format(loss_train.item()),
        "acc_train: {:.4f}".format(acc_train.item()),
        "loss_val: {:.4f}".format(loss_val.item()),
        "acc_val: {:.4f}".format(acc_val.item()),
        "time: {:.4f}s".format(time.time() - t),
    )

    # Return the values to be stored
    return loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item()


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = loader.accuracy(output[idx_test], labels[idx_test])
    print(
        "Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()),
    )

    return loss_test.item(), acc_test.item()


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    # Get training metrics and store them
    loss_train, acc_train, loss_val, acc_val = train(epoch)

    # Append values to lists
    train_losses.append(loss_train)
    val_losses.append(loss_val)
    train_accs.append(acc_train)
    val_accs.append(acc_val)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test_loss, test_acc = test()

# Plot learning curves AFTER training is complete
plt.figure(figsize=(15, 5))

# Plot loss curves
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Train Loss", color="blue")
plt.plot(val_losses, label="Val Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# Plot accuracy curves
plt.subplot(1, 3, 2)
plt.plot(train_accs, label="Train Acc", color="blue")
plt.plot(val_accs, label="Val Acc", color="red")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

# Plot both on same scale for comparison
plt.subplot(1, 3, 3)
plt.plot(train_losses, label="Train Loss", linestyle="--", alpha=0.7)
plt.plot(val_losses, label="Val Loss", linestyle="--", alpha=0.7)
# Normalize accuracy to same scale as loss for visualization
if max(train_accs) > 0 and max(train_losses) > 0:
    plt.plot(
        [acc / max(train_accs) * max(train_losses) for acc in train_accs],
        label="Train Acc (scaled)",
        alpha=0.7,
    )
    plt.plot(
        [acc / max(val_accs) * max(val_losses) for acc in val_accs],
        label="Val Acc (scaled)",
        alpha=0.7,
    )
plt.xlabel("Epoch")
plt.ylabel("Normalized Values")
plt.title("Training Progress Overview")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{args.dataset}_learning_curves.png", dpi=300, bbox_inches="tight")
plt.show()

with open(f"{args.dataset}_loss_data.json", "w") as f:
    json_data = json.dumps(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
    )
    f.write(json_data)

# Print final results summary
print(f"\n=== Final Results for {args.dataset.upper()} Dataset ===")
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
print(f"Final Validation Loss: {val_losses[-1]:.4f}")
print(f"Final Validation Accuracy: {val_accs[-1]:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(
    f"Best Validation Accuracy: {max(val_accs):.4f} at epoch {val_accs.index(max(val_accs))+1}"
)


# python train.py --dataset cora --epochs 200 --lr 0.01
# python train.py --dataset cora --epochs 200 --lr 0.01 --no-cuda
