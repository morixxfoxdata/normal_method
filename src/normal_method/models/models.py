import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

# import wandb
from src.normal_method.data import mnist_total

# 先ほど定義したSSIMクラスをインポートしたと仮定します
# from src.normal_method.models.ssim_model import SSIM
from src.normal_method.speckle.prediction import (
    inv_hadamard,
    speckle_noise_calculation,
)
from src.normal_method.visualization.display import image_display

# wandb.login()

# wandb.init(project="speckle")


def standardization(data: np.ndarray) -> np.ndarray:
    """
    標準化
    """
    return (data - data.mean()) / data.std()


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


class Net_version_2(nn.Module):
    def __init__(self):
        super(Net_version_2, self).__init__()
        self.fc1 = nn.Linear(500, 250)
        self.fc2 = nn.Linear(250, 128)
        self.fc3 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def get_class_name(self):
        return self.__class__.__name__


class Net_version_5(nn.Module):
    def __init__(self):
        super(Net_version_5, self).__init__()
        self.fc1 = nn.Linear(500, 250)
        self.fc2 = nn.Linear(250, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class Net_version_4(nn.Module):
    def __init__(self):
        super(Net_version_4, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def get_class_name(self):
        return self.__class__.__name__


class Net_version_3(nn.Module):
    def __init__(self):
        super(Net_version_3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64))

    def forward(self, x):
        encoded = self.encoder(x)
        logits = self.decoder(encoded)
        return logits

    def get_binary_output(self, x):
        logits = self(x)
        return torch.sign(logits)  # -1 または 1 を返す

    def get_class_name(self):
        return self.__class__.__name__


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
        # print(f"reconstructed_x shape: {reconstructed_x.shape}")
        # Sをかけた結果
        out = reconstructed_x.view(-1, reconstructed_x.size(0))
        out = torch.mm(out, S)
        predicted_y = out.view(out.size(1))
        # print(f"predicted_y shape: {predicted_y.shape}")
        # print(f"y_observed shape: {y_observed.shape}")
        # predicted_y = torch.mm(reconstructed_x, S.t())
        loss = criterion(predicted_y, y_observed)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        # wandb.log({"epoch": epoch, "loss": loss.item()})
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
    selected_model = Net_version_5()
    # 学習画像枚数
    num_images = 10
    # 画像ごとのエポック数
    num_epochs = 10000
    # 利用スペックル
    # selected_speckle = S
    # 標準化の有無
    normalized = False
    learning_rate = 1 * 1e-4
    XX, yy = mnist_total()
    # S_norm_stand = standardization(S_norm)
    speckle = S_norm.T
    print(yy.shape)
    print(speckle.shape)
    # XX_stand = standardization(XX)
    # yy_stand = standardization(yy)
    # wandbに設定をログ
    # wandb.config.update(
    #     {
    #         "speckle_alpha": speckle_alpha,
    #         "num_images": num_images,
    #         "num_epochs": num_epochs,
    #         "learning_rate": learning_rate,
    #         "model": selected_model.get_class_name(),
    #     }
    # )
    """
    訓練
    """
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
    mse = mean_squared_error(XX, nd_recon)
    # mse_bin = mean_squared_error(XX, nd_binary)
    print(f"Average reconstruction MSE: {mse:.4f}")
    print(nd_recon.min(), nd_recon.max())
    # print(nd_binary.min(), nd_binary.max())
    # np.save("data/processed/reconstructed_signals.npy", nd_recon)
    image_display(j=8, xx=XX, yy=nd_recon, size=8)
    # wandbに最終結果をログ
    # wandb.log(
    #     {"final_average_loss": np.mean([loss[-1] for loss in loss_history]), "mse": mse}
    # )
    # wandb.finish()
