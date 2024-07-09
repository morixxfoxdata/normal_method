import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

from src.normal_method.data import mnist_total
from src.normal_method.speckle.prediction import (
    inv_hadamard,
    speckle_noise_calculation,
)
from src.normal_method.visualization.display import image_display


class Net_cnn_ver1(nn.Module):
    def __init__(self):
        super(Net_cnn_ver1, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 123, 256), nn.Tanh(), nn.Linear(256, 64), nn.Tanh()
        )

    def forward(self, x):
        # Input x has shape (500,)
        x = x.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, 500)

        # Apply convolutional layers
        x = self.conv_layers(x)

        # Flatten the output
        x = x.view(1, -1)

        # Apply fully connected layers
        x = self.fc_layers(x)

        return x.squeeze(0)  # Output shape: (64,)


class Net_cnn_ver2(nn.Module):
    def __init__(self):
        super(Net_cnn_ver2, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 124, 256), nn.Tanh(), nn.Linear(256, 64), nn.Tanh()
        )

    def forward(self, x):
        # Input x has shape (500,)
        x = x.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, 500)

        # Apply convolutional layers
        x = self.conv_layers(x)

        # Flatten the output
        x = x.view(1, -1)

        # Apply fully connected layers
        x = self.fc_layers(x)

        return x.squeeze(0)  # Output shape: (64,)


class Net_rnn_ver1(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=64):
        super(Net_rnn_ver1, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer for output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        # Input x has shape (500,)
        x = x.unsqueeze(0).unsqueeze(2)  # Shape becomes (1, 500, 1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (1, 500, hidden_size)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]

        # Pass through the fully connected layers
        out = self.fc(out)

        return out.squeeze(0)  # Output shape: (64,)


def training_network(
    model: nn.Module,
    S: torch.Tensor,
    y_observed: torch.Tensor,
    num_epochs: int,
    learning_rate: float = 1 * 1e-5,
) -> Tuple[List[float], np.ndarray]:
    """
    Args:
        model: 訓練するモデル
        S: スペックル
        y_observed: 観測されたデータ
        num_epochs: 訓練エポック数
        learning_rate: 学習率

    Returns:
        train_loss_list: 各エポックの訓練損失
        train_result: 訓練後のモデル出力
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    y_observed = y_observed.to(device)
    S = S.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed_x = model(y_observed)
        out = reconstructed_x.view(-1, reconstructed_x.size(0))
        out = torch.mm(out, S)
        predicted_y = out.view(out.size(1))
        loss = criterion(predicted_y, y_observed)
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
    model,
    num_images: int,
    num_epochs: int,
    data_y: np.ndarray,
    S: np.ndarray,
    normalized,
    learning_rate,
):
    """
    メイン訓練ループ

    Args:
        num_images: 画像枚数
        num_epochs: 各画像のエポック数
        data_y: 観測データY
        S: スペックル
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    S_tensor = torch.tensor(S).float().to(device)
    # model = Net_version_1()
    model = model

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
            learning_rate=learning_rate,
        )
        loss_total.append(loss_list)
        reconstructed_total.append(reconstructed_x.cpu().numpy())
        elapsed_time = time.time() - start_time
        # wandb.log(
        #     {"iteration": i + 1, "final_loss": loss_list[-1], "time": elapsed_time}
        # )
        # if i == 0 or (i + 1) % 100 == 0:
        print(
            f"Iteration: {i+1}/{num_images}, Final Loss: {loss_list[-1]:.4f}, Time: {elapsed_time}"
        )
    return loss_total, reconstructed_total


if __name__ == "__main__":
    S_hd = inv_hadamard()
    ### スペックル切り替え
    lambda1 = 10
    lambda2 = 10
    speckle_alpha = 1
    # S_org = Original_pred(lambda1=lambda1, lambda2=lambda2)
    S_norm = speckle_noise_calculation(S_hd, alpha=speckle_alpha)
    """
    パラメータ、データ設定
    """
    # 利用モデル
    selected_model = Net_rnn_ver1()
    # 学習画像枚数
    num_images = 10
    # 画像ごとのエポック数
    num_epochs = 1000
    # 利用スペックル
    # selected_speckle = S
    # 標準化の有無
    normalized = False
    learning_rate = 1 * 1e-5
    XX, yy = mnist_total()
    # S_norm_stand = standardization(S_norm)
    speckle = S_norm.T
    print("input shape:", yy.shape)
    print("Speckle shape:", speckle.shape)
    print("output shape:", XX.shape)
    loss_history, reconstructed_signals = main_train(
        selected_model,
        num_images,
        num_epochs,
        yy,
        speckle,
        normalized,
        learning_rate,
    )
    print("Training completed.")
    print(f"Final average loss: {np.mean([loss[-1] for loss in loss_history]):.4f}")
    nd_recon = np.array(reconstructed_signals)
    nd_loss = np.array(loss_history)
    # 再構成の精度評価
    mse_val = mean_squared_error(XX, nd_recon)
    print(f"Average reconstruction MSE: {mse_val:.4f}")
    print(f"Min value:{nd_recon.min()}, Max value:{nd_recon.max()}")
    image_display(j=8, xx=XX, yy=nd_recon, size=8)
