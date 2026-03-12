"""GNN training script.

Trains the BasisRec GNN on labelled data from QCArchive and saves
the model weights to data/pretrained_gnn.pt.

Usage:
    python scripts/train_gnn.py
    python scripts/train_gnn.py --epochs 100 --lr 1e-3 --output my_model.pt

The training set is assembled by:
    1. Downloading small molecules from QCArchive.
    2. Running calculations at cc-pVTZ (reference) and several
       smaller basis sets.
    3. Labelling each atom with the minimum basis that recovers
       99% of the cc-pVTZ energy contribution.

This script requires: torch, torch-geometric, qcportal, pyscf.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BasisRec GNN model.")
    p.add_argument("--epochs",  type=int,   default=50,             help="Training epochs")
    p.add_argument("--lr",      type=float, default=1e-3,           help="Learning rate")
    p.add_argument("--batch",   type=int,   default=32,             help="Batch size")
    p.add_argument("--hidden",  type=int,   default=128,            help="GNN hidden dim")
    p.add_argument("--layers",  type=int,   default=3,              help="GNN message-passing layers")
    p.add_argument("--data",    type=Path,  default=Path("data/training_set.pkl"),
                   help="Path to training data pickle")
    p.add_argument("--output",  type=Path,  default=Path("data/pretrained_gnn.pt"),
                   help="Output path for trained weights")
    p.add_argument("--seed",    type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        logger.error("torch is required for training. pip install torch")
        sys.exit(1)

    torch.manual_seed(args.seed)

    if not args.data.exists():
        logger.error("Training data not found at '%s'.", args.data)
        logger.info("Generate training data first with scripts/build_training_set.py")
        sys.exit(1)

    import pickle
    logger.info("Loading training data from '%s'…", args.data)
    with open(args.data, "rb") as fh:
        dataset = pickle.load(fh)

    # dataset is a list of (feature_vector, label_index) tuples
    from basisrec.gnn_model import (
        _GNNArchitecture, _NODE_FEATURE_DIM, BASIS_CLASSES
    )

    model = _GNNArchitecture(
        node_features=_NODE_FEATURE_DIM,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=len(BASIS_CLASSES),
        dropout=0.1,
    )

    X = torch.tensor([d[0] for d in dataset], dtype=torch.float32)
    y = torch.tensor([d[1] for d in dataset], dtype=torch.long)

    # Train/val split 90/10
    n_train = int(0.9 * len(dataset))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    optimizer = optim.Adam(model._net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model._net.train()
        # Mini-batch loop
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), args.batch):
            idx = perm[i:i + args.batch]
            optimizer.zero_grad()
            out = model(X_train[idx])
            loss = criterion(out, y_train[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        # Validation
        model._net.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_preds = val_out.argmax(dim=-1)
            val_acc = (val_preds == y_val).float().mean().item()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  loss=%.4f  val_acc=%.3f",
                epoch, args.epochs, total_loss / len(X_train), val_acc,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model._net.state_dict(), args.output)

    logger.info("Training complete. Best val accuracy: %.3f", best_val_acc)
    logger.info("Model saved to '%s'.", args.output)


if __name__ == "__main__":
    main()
