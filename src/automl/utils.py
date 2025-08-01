from typing import Any
import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_mean_std(dataset_class: Any):
    """Calculate the mean and standard deviation of the entire image dataset."""
    mean = 0.
    std = 0.
    total_images_count = 0

    dataset = dataset_class(
        root="./data",
        split='train',
        download=True,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std
