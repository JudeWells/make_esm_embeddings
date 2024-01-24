import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def get_labels(experiment_name):
    csv_path = f"../DMS_Datasets/{experiment_name}.csv"
    df = pd.read_csv(csv_path)
    return df.DMS_score.values


def make_data_loader(data_dir):
    for fpath in glob.glob(f"{data_dir}/embeddings*.npy"):
        print(fpath)
        x = np.load(fpath)
    experiment_name = data_dir.split('/')[-1]
    y = get_labels(experiment_name)
    y = y[:x.shape[0]]
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False)

    shape_dict = {
        'num_sequences': len(train_dataset),
        'sequence_length': x_train.shape[1],
        'embedding_dimension': x_train.shape[2]
    }
    return train_loader, test_loader, shape_dict


class RegressionNN(nn.Module):
    def __init__(self, sequence_length, embedding_dim_plm, embedding_dim_out=10):
        super(RegressionNN, self).__init__()
        self.ffn1 = nn.Linear(embedding_dim_plm, embedding_dim_out)
        self.ffn2 = nn.Linear(embedding_dim_out, 1)

    def forward(self, x):
        x = torch.relu(self.ffn1(x))
        # sum over the sequence length dimension
        x = torch.sum(x, dim=1)
        x = self.ffn2(x)
        return x


if __name__ == "__main__":
    data_dir = '../DMS_embeddings/PIN1_HUMAN_Tsuboyama_2023_1I6C'
    train_loader, test_loader, shape_dict = make_data_loader(data_dir)

    model = RegressionNN(shape_dict['sequence_length'], shape_dict['embedding_dimension'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  # number of epochs
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.view(-1, 1))
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        test_loss = 0
        pred_vals = []
        true_vals = []
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target.view(-1, 1)).item()
                pred_vals.extend(list(output.numpy().reshape(-1)))

        test_loss /= len(test_loader.dataset)
        print(f"Epoch: {epoch + 1}, Test Loss: {test_loss:.4f}")

    bp=1
    pass