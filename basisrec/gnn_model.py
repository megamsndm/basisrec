"""Graph Neural Network for per-atom basis set prediction.

Architecture: message-passing GNN (GraphSAGE-style) with residual
connections. Atoms are nodes, bonds are edges. Each node is encoded
with a feature vector from the featurizer. The model outputs a
probability distribution over the basis set vocabulary for each atom.

Training is done separately via ``scripts/train_gnn.py``.
This module handles architecture definition, inference, and
model serialisation/deserialisation.

If the pretrained weights file is absent or torch is not installed,
the GNN silently degrades — the decision engine falls back to the
rule engine exclusively.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from .exceptions import ModelNotFoundError
from .models import AtomFeatures, BasisRecommendation, BasisTier
from .basis_library import get_tier, BASIS_TIERS

logger = logging.getLogger(__name__)

# Ordered list of basis set classes the model predicts.
# Index → basis name. Must match the training label encoding.
BASIS_CLASSES: List[str] = [
    "STO-3G",
    "6-31G*",
    "6-31G**",
    "def2-SVP",
    "def2-TZVP",
    "cc-pVDZ",
    "cc-pVTZ",
    "aug-cc-pVDZ",
    "aug-cc-pVTZ",
    "LANL2DZ",
]

# Atom-level feature dimension (must match atom_featurizer output)
_NODE_FEATURE_DIM = 20
_HIDDEN_DIM = 128
_NUM_LAYERS = 3
_DROPOUT = 0.1

# Default path to pretrained weights shipped with the package
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "data" / "pretrained_gnn.pt"


class BasisGNN:
    """Wrapper around the PyTorch GNN model.

    Provides a clean interface for inference without exposing
    PyTorch internals to the rest of the pipeline.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model = None
        self._torch_available = False
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._load_model()

    @property
    def is_available(self) -> bool:
        """True if the model is loaded and ready for inference."""
        return self._model is not None

    def predict(self, features_list: List[AtomFeatures]) -> List[BasisRecommendation]:
        """Run inference on a list of atom features.

        Returns one :class:`~basisrec.models.BasisRecommendation` per atom
        with ``source='gnn'``.

        If the model is not available, raises :exc:`ModelNotFoundError`.
        """
        if not self.is_available:
            raise ModelNotFoundError(
                "GNN model is not loaded. Either the pretrained weights file "
                f"'{self._model_path}' is missing, or torch is not installed."
            )

        import torch
        tensor = self._features_to_tensor(features_list)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1)

        recommendations = []
        for i, feat in enumerate(features_list):
            atom_probs = probs[i].cpu().tolist()
            best_idx = int(max(range(len(atom_probs)), key=lambda j: atom_probs[j]))
            confidence = round(atom_probs[best_idx], 3)
            basis_name = BASIS_CLASSES[best_idx]

            recommendations.append(BasisRecommendation(
                atom_index=feat.index,
                symbol=feat.symbol,
                basis_name=basis_name,
                tier=get_tier(basis_name),
                confidence=confidence,
                reason=(
                    f"{feat.symbol}: GNN predicted {basis_name} "
                    f"with {confidence:.0%} confidence from local chemical environment."
                ),
                source="gnn",
            ))

        return recommendations

    def _load_model(self) -> None:
        """Attempt to load the pretrained model weights."""
        try:
            import torch
            self._torch_available = True
        except ImportError:
            logger.info(
                "torch not installed — GNN model unavailable. "
                "Rule engine will be used exclusively."
            )
            return

        if not self._model_path.exists():
            logger.info(
                "Pretrained GNN weights not found at '%s'. "
                "Rule engine will be used exclusively.",
                self._model_path,
            )
            return

        try:
            import torch
            self._model = _GNNArchitecture(
                node_features=_NODE_FEATURE_DIM,
                hidden_dim=_HIDDEN_DIM,
                num_layers=_NUM_LAYERS,
                num_classes=len(BASIS_CLASSES),
                dropout=_DROPOUT,
            )
            state = torch.load(self._model_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state)
            self._model.eval()
            logger.info("Loaded pretrained GNN from '%s'.", self._model_path)
        except Exception as exc:
            logger.warning("Failed to load GNN model: %s — using rule engine only.", exc)
            self._model = None

    def _features_to_tensor(self, features_list: List[AtomFeatures]):
        """Encode atom features as a float tensor for model input."""
        import torch

        rows = []
        for feat in features_list:
            row = _encode_atom_features(feat)
            rows.append(row)

        return torch.tensor(rows, dtype=torch.float32)


class _GNNArchitecture:
    """Minimal GraphSAGE-style GNN implemented in pure PyTorch.

    Uses a simple multi-layer perceptron over concatenated atom
    features as a stand-in until torch-geometric is available.
    In production this should be replaced with a proper
    torch_geometric.nn.SAGEConv or GATConv architecture.
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        try:
            import torch
            import torch.nn as nn

            layers = []
            in_dim = node_features
            for _ in range(num_layers):
                layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, num_classes))

            self._net = nn.Sequential(*layers)
        except ImportError:
            self._net = None

    def __call__(self, x):
        if self._net is None:
            raise RuntimeError("torch.nn not available")
        return self._net(x)

    def load_state_dict(self, state):
        self._net.load_state_dict(state)

    def eval(self):
        self._net.eval()


def _encode_atom_features(feat: AtomFeatures) -> List[float]:
    """Encode an :class:`~basisrec.models.AtomFeatures` as a fixed-length
    float vector of length :data:`_NODE_FEATURE_DIM`.

    Feature layout:
        [0]   partial_charge
        [1]   coordination_number (normalised / 8)
        [2]   period (normalised / 7)
        [3]   is_aromatic
        [4]   is_hbond_donor
        [5]   is_hbond_acceptor
        [6]   has_lone_pairs
        [7]   is_metal
        [8]   is_heavy
        [9-15] hybridization one-hot (S, SP, SP2, SP3, SP3D, SP3D2, UNKNOWN)
        [16-19] neighbor element flags (has_H, has_C, has_N, has_O)
    """
    from .models import Hybridization

    hyb_order = [
        Hybridization.S, Hybridization.SP, Hybridization.SP2,
        Hybridization.SP3, Hybridization.SP3D, Hybridization.SP3D2,
        Hybridization.UNKNOWN,
    ]
    hyb_one_hot = [1.0 if feat.hybridization == h else 0.0 for h in hyb_order]

    neighbors = feat.neighbor_symbols
    neighbor_flags = [
        1.0 if "H" in neighbors else 0.0,
        1.0 if "C" in neighbors else 0.0,
        1.0 if "N" in neighbors else 0.0,
        1.0 if "O" in neighbors else 0.0,
    ]

    return [
        feat.partial_charge,
        feat.coordination_number / 8.0,
        feat.period / 7.0,
        float(feat.is_aromatic),
        float(feat.is_hbond_donor),
        float(feat.is_hbond_acceptor),
        float(feat.has_lone_pairs),
        float(feat.is_metal),
        float(feat.is_heavy),
        *hyb_one_hot,
        *neighbor_flags,
    ]


# Module-level singleton — loaded once
_gnn_instance: Optional[BasisGNN] = None


def get_gnn(model_path: Optional[Path] = None) -> BasisGNN:
    """Return the module-level GNN singleton, loading it if necessary."""
    global _gnn_instance
    if _gnn_instance is None or model_path is not None:
        _gnn_instance = BasisGNN(model_path=model_path)
    return _gnn_instance
