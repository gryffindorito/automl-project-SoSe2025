import os
from torch.utils.data import DataLoader, random_split
from src.automl.dataloader_utils import FreiburgDataset


def get_dataset_dataloaders(dataset_name, root='automl_data', batch_size=64, val_split=0.2):
    """
    Returns train, val, and test DataLoaders for a given dataset.

    Args:
        dataset_name (str): Name of the dataset folder (e.g. 'flowers').
        root (str): Root directory containing all datasets.
        batch_size (int): Batch size for DataLoaders.
        val_split (float): Fraction of training set to use as validation.

    Returns:
        train_loader, val_loader, test_loader (torch.utils.data.DataLoader)
    """
    dataset_path = os.path.join(root, dataset_name)
    
    # Create full training and test datasets
    train_dataset = FreiburgDataset(dataset_path, split='train')
    test_dataset = FreiburgDataset(dataset_path, split='test')

    # Split training into train/val
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"✅ Dataset split complete.\nTrain: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
