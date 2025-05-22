from train import LogisticRegression, MNISTDataset
import torch
from torch.utils.data import DataLoader


def validate(model, val_loader):
    model.eval()
    predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            outputs = outputs.softmax(dim=-1)
            _, predicted = torch.max(outputs, dim=-1)
            predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    predictions = torch.tensor(predictions)
    all_labels = torch.tensor(all_labels)
    accuracy = (predictions == all_labels).sum().item() / len(all_labels)
    print(f"Validation accuracy: {accuracy}")


if __name__ == "__main__":
    # load model
    model = LogisticRegression()
    model.load_state_dict(torch.load("ckpts/model.pt", weights_only=True))
    # load data
    val_dataset = MNISTDataset("data/mnist-train7k-val3k/val.csv")
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    # validate
    validate(model, val_loader)
