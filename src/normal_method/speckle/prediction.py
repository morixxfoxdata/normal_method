import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.normal_method.data import hadamard_total, random_total


def inv_hadamard():
    X_hadamard, y_hadamard = hadamard_total()
    n = int(X_hadamard.shape[0])
    hadamard_sp = np.dot(y_hadamard.T, X_hadamard.T) / n
    # hadamard_sp.shape: (500, 64)
    return hadamard_sp


def random_pattern_split():
    # X_random.shape: (500, 64)
    # y_random.shape: (500, 500)
    X_random, y_random = random_total()
    X_train, X_test, y_train, y_test = train_test_split(
        X_random, y_random, test_size=0.1, shuffle=False
    )
    # X_train (450, 64), X_test (50, 64), y_train (450, 500), y_test (50, 500)
    X_train1 = X_train[0 : int(X_train.shape[0] / 2), :]
    X_train2 = X_train[int(X_train.shape[0] / 2) :, :]

    y_train1 = y_train[0 : int(X_train.shape[0] / 2), :]
    y_train2 = y_train[int(X_train.shape[0] / 2) :, :]
    return X_train1, X_train2, y_train1, y_train2


def speckle_noise_calculation(S, alpha=1):
    _, X_train2, _, y_train2 = random_pattern_split()
    delta = y_train2 - np.dot(X_train2, S.T)
    delta_Ridge = Ridge(alpha=alpha)
    delta_Ridge.fit(X_train2, delta)
    delta_ridge_coef = delta_Ridge.coef_
    predicted_speckle = S + delta_ridge_coef
    # speckle.shape: (500, 64)
    return predicted_speckle


class CustomLoss(nn.Module):
    def __init__(self, S_hd, lambda1, lambda2):
        super(CustomLoss, self).__init__()
        self.S_hd = torch.tensor(S_hd, dtype=torch.float32)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.Delta_S = nn.Parameter(torch.zeros_like(self.S_hd))

    def forward(self, y_true, y_pred, x):
        error_term = torch.mean((y_true - (self.S_hd - self.Delta_S) @ x.T) ** 2)
        l2_term = self.lambda1 * torch.mean(self.Delta_S**2)
        l1_term = self.lambda2 * torch.mean(torch.abs(x))
        return error_term + l2_term + l1_term


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        return self.linear(x)


def Original_pred(lambda1, lambda2):
    X_train1, X_train2, y_train1, y_train2 = random_pattern_split()
    y_train1 = y_train1.T
    X_train1 = torch.tensor(X_train1, dtype=torch.float32)
    y_train1 = torch.tensor(y_train1, dtype=torch.float32)
    S_hd = inv_hadamard()
    model = SimpleLinearModel()
    # print(S_hd.shape)
    X_random, y_random = random_total()
    X_train, X_test, y_train, y_test = train_test_split(
        X_random, y_random, test_size=0.1, shuffle=False
    )
    # parameter
    lambda1 = lambda1
    lambda2 = lambda2

    # 損失関数の定義
    criterion = CustomLoss(S_hd, lambda1, lambda2)
    optimizer = optim.Adam(list(model.parameters()) + [criterion.Delta_S], lr=0.00005)

    # モデルの訓練
    num_epochs = 100
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train1)
        loss = criterion(y_train1, y_pred, X_train1)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {1000*loss.item():.6f}")
    # 予測
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train1)
    del_S = criterion.Delta_S.detach().numpy()
    return S_hd - del_S, loss_list
    # # print("Predicted values:", y_pred.view(-1).numpy())
    # # print("Estimated Delta_S:", del_S.shape)

    # display_image_random(4, S_hd, X_test