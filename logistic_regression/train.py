import pandas as pd
import torch
from torch.utils.data import Dataset, dataloader
import torch.nn as nn
import torch.optim as optim


class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = torch.tensor(self.data.iloc[idx, 0], dtype=torch.long)
        image = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float32)
        # normalize(important)
        image = image / 255.0
        return image, label


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(784, 10)

    def forward(self, x):
        x = self.layer(x)
        return x


if __name__ == "__main__":
    # set hyperparameters
    num_epochs = 20
    batch_size = 128
    learning_rate = 1

    # load data
    train_dataset = MNISTDataset("data/mnist-train7k-val3k/train.csv")
    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    # create model
    model = LogisticRegression()
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # train model
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update running loss
            running_loss += loss.item() * images.size(0)
        # print loss
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # save model
    torch.save(model.state_dict(), "ckpts/model.pt")
