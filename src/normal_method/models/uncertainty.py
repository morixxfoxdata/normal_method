import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from src.normal_method.data import mnist_total
from src.normal_method.speckle.prediction import inv_hadamard, speckle_noise_calculation
from src.normal_method.visualization.display import image_display


class Net_version_1(nn.Module):
    def __init__(self):
        super(Net_version_1, self).__init__()
        self.fc1 = nn.Linear(500, 250)
        self.fc2 = nn.Linear(250, 64)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

    def get_class_name(self):
        return self.__class__.__name__


class UncertaintyAwareModel(nn.Module):
    def __init__(self, base_model, dropout_rate=0.1):
        super(UncertaintyAwareModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, Y):
        X_prime = self.base_model(Y)
        X_prime = self.dropout(X_prime)
        return X_prime

    def predict_with_uncertainty(self, Y, S, num_samples=100):
        self.train()
        X_predictions = []
        Y_predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                X_prime = self(Y)
                Y_prime = torch.mm(X_prime, S)
                X_predictions.append(X_prime)
                Y_predictions.append(Y_prime)

        mean_X = torch.mean(torch.stack(X_predictions), dim=0)
        std_X = torch.std(torch.stack(X_predictions), dim=0)
        mean_Y = torch.mean(torch.stack(Y_predictions), dim=0)
        std_Y = torch.std(torch.stack(Y_predictions), dim=0)
        return mean_X, std_X, mean_Y, std_Y


def training_network(
    model: UncertaintyAwareModel,
    S: torch.Tensor,
    y_observed: torch.Tensor,
    num_epochs: int,
    learning_rate: float = 1e-5,
    num_samples: int = 10,
) -> Tuple[List[float], torch.Tensor]:
    device = y_observed.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    loss_list = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        loss = 0
        for _ in range(num_samples):
            reconstructed_x = model(y_observed)

            y_prime = torch.mm(reconstructed_x.view(-1, reconstructed_x.size(0)), S)
            loss += criterion(y_prime.view(y_prime.size(1)), y_observed)

        loss /= num_samples
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch: [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        reconstructed_x = model(y_observed)

    return loss_list, reconstructed_x


def main_train(
    base_model,
    num_images: int,
    num_epochs: int,
    data_y: np.ndarray,
    S: np.ndarray,
    normalized: bool,
    learing_rate: float,
    dropout_rate: float = 0.1,
    num_samples: int = 10,
) -> Tuple[List[List[float]], List[np.ndarray]]:
    """
    訓練ループ

    Args:
        base_model: ベースモデル
        num_images: 画像枚数
        num_epochs: 各画像のエポック数
        data_y: 観測データY
        S: スペックル
        normalized: 正規化
        learing_rate: 学習率
        dropout_rate: ドロップアウト率
        num_samples: モンテカルロサンプル数

    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    S_tensor = torch.tensor(S).float().to(device)
    model = UncertaintyAwareModel(base_model, dropout_rate)

    loss_total = []
    reconstructed_total = []
    start_time = time.time()

    for i in range(num_images):
        y_observed = torch.tensor(data_y[i]).float().to(device)
        loss_list, reconstructed_x = training_network(
            model,
            S_tensor,
            y_observed,
            num_epochs=num_epochs,
            learning_rate=learing_rate,
            num_samples=num_samples,
        )
        loss_total.append(loss_list)
        reconstructed_total.append(reconstructed_x.cpu().numpy())

        elapsed_time = time.time() - start_time

        print(
            f"Iteration: {i+1}/{num_images}, Final Loss: {loss_list[-1]:.4f}, Time: {elapsed_time}"
        )
    return loss_total, reconstructed_total


if __name__ == "__main__":
    base_model = Net_version_1()
    S_hd = inv_hadamard()
    speckle_alpha = 1
    S_norm = speckle_noise_calculation(S_hd, alpha=speckle_alpha)
    uncertainty_model = UncertaintyAwareModel(base_model)
    XX, yy = mnist_total()
    speckle = S_norm.T
    learning_rate = 2e-5
    dropout_rate = 0.01
    num_samples = 100
    num_images = 10
    num_epochs = 10000
    normalized = False
    # # 訓練時
    # X_prime = uncertainty_model(Y)
    # Y_prime = torch.mm(X_prime, S)
    print(S_hd.shape)  # (500, 64)
    print(S_norm.shape)  # (500, 64)
    print(speckle.shape)  # (64, 500
    print(XX.shape)  # (10, 64)
    print(yy.shape)  # (10, 500)
    loss_total, reconstructed_total = main_train(
        base_model,
        num_images,
        num_epochs,
        yy,
        speckle,
        normalized,
        learning_rate,
        dropout_rate,
        num_samples,
    )
    print("Training completed.")
    print(f"Final average loss: {np.mean([loss[-1] for loss in loss_total]):.4f}")
    nd_recon = np.array(reconstructed_total)
    nd_loss = np.array(loss_total)
    mse_val = mean_squared_error(XX, nd_recon)
    print(f"Average reconstruction MSE: {mse_val:.4f}")
    print(nd_recon.min(), nd_recon.max())
    image_display(j=8, xx=XX, yy=nd_recon, size=8)
