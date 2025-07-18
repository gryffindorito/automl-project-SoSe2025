import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class FreiburgDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "test"]
        self.transform = transform
        self.split = split
        self.root = root

        # Load CSV and file list
        df = pd.read_csv(os.path.join(root, f"{split}.csv"))
        self.image_files = df["image_file_name"].tolist()
        self.labels = df["label"].tolist()

        self.image_dir = os.path.join(root, f"images_{split}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(dataset_name, data_dir="data", batch_size=64, num_workers=2):
    dataset_root = os.path.join(data_dir, dataset_name)

    # Define preprocessing per dataset
    if dataset_name == "fashion":
        input_size = 28
        channels = 1
        mean, std = [0.5], [0.5]
    elif dataset_name == "flowers":
        input_size = 224
        channels = 3
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif dataset_name == "emotions":
        input_size = 48
        channels = 1
        mean, std = [0.5], [0.5]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = FreiburgDataset(dataset_root, split="train", transform=transform)
    test_dataset = FreiburgDataset(dataset_root, split="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        "train": train_loader,
        "test": test_loader
    }
