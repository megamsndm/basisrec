"""Custom exceptions for BasisRec."""


class BasisRecError(Exception):
    """Base exception for all BasisRec errors."""


class ParseError(BasisRecError):
    """Raised when the XYZ file cannot be parsed."""


class FeaturizationError(BasisRecError):
    """Raised when atom featurization fails."""


class UnsupportedElementError(BasisRecError):
    """Raised when an element has no available basis set."""

    def __init__(self, symbol: str, basis_name: str) -> None:
        self.symbol = symbol
        self.basis_name = basis_name
        super().__init__(
            f"Basis set '{basis_name}' does not cover element '{symbol}'. "
            f"Use bse.filter_basis_sets(elements=['{symbol}']) to find alternatives."
        )


class ModelNotFoundError(BasisRecError):
    """Raised when the pretrained GNN weights file is missing."""


class ORCAWriterError(BasisRecError):
    """Raised when ORCA input file generation fails."""
