import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train(model, train_loader, val_loader, device, epochs=20):
    """
    Trains a model and returns validation accuracy per epoch.
    Expects train_loader and val_loader to be actual DataLoader objects.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    val_accs = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images, labels = batch
            # print(f"TYPE OF labels: {type(labels)} | VALUE: {labels}")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                # print(f"TYPE OF labels: {type(labels)} | VALUE: {labels}")
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        val_accs.append(acc)
        print(f"Epoch {epoch+1}: Val Accuracy = {acc:.4f}")

    return val_accs
