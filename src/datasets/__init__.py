"""Dataset adapters, validation, and deterministic loaders."""

from .adapters import DatasetDescription, ImageRecord, inspect_dataset
from .data_module import build_dataloaders
from .splits import make_splits, validate_split_integrity

__all__ = ["DatasetDescription", "ImageRecord", "inspect_dataset", "build_dataloaders", "make_splits", "validate_split_integrity"]