import torch
import numpy as np
from typing import Callable
from typing_extensions import Self

class CIFARDataset(torch.utils.data.Dataset):
    def __init__(
        self: Self,
        filepath: str = "cifar-10-python.npz",
        transforms: Callable = None,
        target_transforms: Callable = None,
    ) -> None:

        super().__init__()

        with np.load(filepath) as data:
            x = np.concatenate((data["x_train"], data["x_test"]))
            y = np.concatenate((data["y_train"], data["y_test"]))

            self.data = x
            self.labels = y

        self.transforms = transforms
        self.target_transforms = target_transforms
    
    def __len__(self: Self) -> int:
        return len(self.labels)

    def __getitem__(self: Self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.data[idx]
        label = self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transforms is not None:
            label = self.target_transforms(label)

        return img, label
