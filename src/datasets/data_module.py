"""PyTorch image datasets and deterministic loaders."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .adapters import DatasetDescription, ImageRecord
from .splits import make_splits
from src.preprocessing import PreprocessingPipeline


class MedicalImageDataset(Dataset):
    def __init__(
        self,
        records: tuple[ImageRecord, ...],
        class_to_index: dict[str, int],
        *,
        image_size: int = 224,
        preprocessing: PreprocessingPipeline | None = None,
        imagenet_normalization: bool = True,
    ) -> None:
        self.records = records
        self.class_to_index = class_to_index
        self.image_size = image_size
        self.preprocessing = preprocessing or PreprocessingPipeline()
        self.imagenet_normalization = imagenet_normalization

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        bgr = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unable to decode image: {record.path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        processed = self.preprocessing(resized)
        tensor = torch.from_numpy(processed).permute(2, 0, 1).float()
        if self.imagenet_normalization:
            mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype)[:, None, None]
            standard_deviation = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype)[:, None, None]
            tensor = (tensor - mean) / standard_deviation
        return tensor, self.class_to_index[record.label]


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    description: DatasetDescription,
    *,
    seed: int = 42,
    batch_size: int = 32,
    workers: int = 0,
    image_size: int = 224,
    preprocessing_method: str = "P0",
    grayscale_compatible: bool = False,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    imagenet_normalization: bool = True,
) -> tuple[dict[str, DataLoader], dict[str, int], dict[str, tuple[ImageRecord, ...]]]:
    if batch_size < 1 or workers < 0 or image_size < 1:
        raise ValueError("batch_size and image_size must be positive; workers cannot be negative.")
    splits = make_splits(description, seed=seed, ratios=ratios)
    class_to_index = {label: index for index, label in enumerate(description.classes)}
    pipeline = PreprocessingPipeline(preprocessing_method, grayscale_compatible)
    loaders = {}
    for split_name, records in splits.items():
        dataset = MedicalImageDataset(
            records,
            class_to_index,
            image_size=image_size,
            preprocessing=pipeline,
            imagenet_normalization=imagenet_normalization,
        )
        generator = torch.Generator().manual_seed(seed)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=_seed_worker,
            generator=generator,
            persistent_workers=workers > 0,
        )
    return loaders, class_to_index, splits