import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error

import wandb
from src.normal_method.data import mnist_total
from src.normal_method.speckle.prediction import (
    inv_hadamard,
    speckle_noise_calculation,
)
from src.normal_method.visualization.display import image_display

wandb.login()

wandb.init(project="speckle")


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


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = (
                self.create_window(self.window_size).to(img1.device).type(img1.dtype)
            )
            self.window = window

        return self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )

    def create_window(self, window_size):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window

    def gaussian(self, window_size, sigma):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        return g / g.sum()

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.ssim_module = SSIM(window_size=window_size)

    def forward(self, img1, img2):
        return 1 - self.ssim_module(img1, img2)


class OptimizedDifferentiableMutualInformationLoss(nn.Module):
    def __init__(self, sigma=0.1, num_samples=1000):
        super().__init__()
        self.sigma = sigma
        self.num_samples = num_samples

    def gaussian_kernel(self, x, y):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-(dist**2) / (2 * self.sigma**2))

    def kde_entropy(self, x):
        batch_size = x.size(0)
        if batch_size > self.num_samples:
            # Random sampling if batch size is too large
            idx = torch.randperm(batch_size)[: self.num_samples]
            x = x[idx]
            batch_size = self.num_samples

        kernel_matrix = self.gaussian_kernel(x, x)
        kernel_sum = (torch.sum(kernel_matrix, dim=1) - 1) / (
            batch_size - 1
        )  # Exclude self
        return -torch.mean(torch.log(kernel_sum + 1e-8))

    def forward(self, predicted_y, y_observed):
        # Ensure inputs are float tensors and reshape
        predicted_y = predicted_y.float().view(-1, 1)
        y_observed = y_observed.float().view(-1, 1)

        # Combine inputs
        joint = torch.cat([y_observed, predicted_y], dim=1)

        # Compute entropies
        h_joint = self.kde_entropy(joint)
        h_y = self.kde_entropy(y_observed)
        h_pred_y = self.kde_entropy(predicted_y)

        # Compute mutual information
        mi = h_y + h_pred_y - h_joint

        # Return negative MI as we want to maximize it
        return -mi


# training_network 関数内での変更
def training_network(
    model: nn.Module,
    S: torch.Tensor,
    y_observed: torch.Tensor,
    num_epochs: int,
    learning_rate: float = 1 * 1e-5,
) -> Tuple[List[float], np.ndarray]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    y_observed = y_observed.to(device)
    S = S.to(device)

    # SSIMLoss をcriterionとして使用
    # criterion = SSIMLoss(window_size=4)  # window_sizeは適宜調整してください
    criterion = OptimizedDifferentiableMutualInformationLoss(
        sigma=0.1, num_samples=1000
    )
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
        wandb.log({"epoch": epoch, "loss": loss.item()})
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

        wandb.log(
            {"iteration": i + 1, "final_loss": loss_list[-1], "time": elapsed_time}
        )
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
    selected_model = Net_version_1()
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
    print(yy.shape)
    print(speckle.shape)
    wandb.config.update(
        {
            "speckle_alpha": speckle_alpha,
            "num_images": num_images,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "model": selected_model.get_class_name(),
        }
    )
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
    print(f"Average reconstruction MSE: {mse:.4f}")
    print(nd_recon.min(), nd_recon.max())
    # np.save("data/processed/reconstructed_signals.npy", nd_recon)
    image_display(j=8, xx=XX, yy=nd_recon, size=8)
    # wandbに最終結果をログ
    wandb.log(
        {"final_average_loss": np.mean([loss[-1] for loss in loss_history]), "mse": mse}
    )
    wandb.finish()
