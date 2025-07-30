import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


class FreiburgDataset(Dataset):
    def __init__(self, root, split="train", transform=None, grayscale=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.grayscale = grayscale

        csv_path = os.path.join(root, f"{split}.csv")
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename_raw = str(self.data.iloc[idx, 1])  # e.g. "008886.jpg"
        label = int(self.data.iloc[idx, 0])         # e.g. 0, 1, ..., 9

        folder = (
            "images_train" if self.split == "train"
            else "images_val" if self.split == "val"
            else "images_test"
        )

        image_path = os.path.join(self.root, folder, filename_raw)
        image = Image.open(image_path)

        if self.grayscale:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(dataset_name: str, root: str = "data", batch_size: int = 64):
    """
    Loads dataloaders using FreiburgDataset (CSV-based).
    If no CSV exists, fallback to ImageFolder (folder-based).
    """
    dataset_root = os.path.join(root, dataset_name)
    is_grayscale = dataset_name == "fashion"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1) if is_grayscale else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])

    train_csv_path = os.path.join(dataset_root, "train.csv")
    val_csv_path = os.path.join(dataset_root, "val.csv")
    test_csv_path = os.path.join(dataset_root, "test.csv")

    if os.path.exists(train_csv_path):
        full_train_dataset = FreiburgDataset(dataset_root, split="train", transform=transform, grayscale=is_grayscale)

        if os.path.exists(val_csv_path):
            train_dataset = full_train_dataset
            val_dataset = FreiburgDataset(dataset_root, split="val", transform=transform, grayscale=is_grayscale)
        else:
            train_len = int(0.8 * len(full_train_dataset))
            val_len = len(full_train_dataset) - train_len
            train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len])

        test_dataset = (
            FreiburgDataset(dataset_root, split="test", transform=transform, grayscale=is_grayscale)
            if os.path.exists(test_csv_path)
            else val_dataset
        )

    else:
        raise FileNotFoundError(f"🚨 Could not find train.csv in {dataset_root}. Cannot use FreiburgDataset or fallback to ImageFolder.")

    print("✅ Dataset split complete.")
    print("Train:", len(train_dataset), "| Val:", len(val_dataset), "| Test:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

