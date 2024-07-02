import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.normal_method.data import random_total
from src.normal_method.speckle.prediction import inv_hadamard

X, y = random_total()
# print(X_random.shape, y_random.shape)   # (500, 64) (500, 500)
# 訓練、検証、テストデータに分割
# 訓練、検証、テストデータに分割
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# PyTorchのテンソルに変換
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class DeltaSModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeltaSModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class CustomLoss(nn.Module):
    def __init__(self, lambda1, lambda2, S0):
        super(CustomLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.S0 = torch.tensor(S0, dtype=torch.float32)

    def forward(self, y_pred, y_true, x, delta_S):
        S0_device = self.S0.to(delta_S.device)
        term1 = torch.sum((y_true - torch.matmul((S0_device + delta_S), x.T).T) ** 2)
        term2 = self.lambda1 * torch.sum(delta_S**2)
        term3 = self.lambda2 * torch.sum(torch.abs(x))
        return term1 + term2 + term3


input_size = X_train.shape[1]  # 64
hidden_size = 128  # 任意の値、調整可能
output_size = y_train.shape[1]  # 500
S0 = inv_hadamard()

model = DeltaSModel(input_size, hidden_size, output_size)

# ハイパーパラメータの設定
lambda1 = 0.1
lambda2 = 0.1

criterion = CustomLoss(lambda1, lambda2, S0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 順伝播
    outputs = model(X_train)
    delta_S = model.fc2.weight  # モデルの重みをΔSとする

    # カスタム損失関数の計算
    loss = criterion(outputs, y_train, X_train, delta_S)

    # 逆伝播と最適化
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # 検証データでのパフォーマンス評価
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val, X_val, delta_S)
        val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
        )

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    delta_S = model.fc2.weight  # モデルの重みをΔSとする
    test_loss = criterion(test_outputs, y_test, X_test, delta_S)
    print(f"Test Loss: {test_loss.item():.4f}")

    # 必要に応じて、さらに詳細な評価を行う
    test_outputs_np = test_outputs.numpy()
    y_test_np = y_test.numpy()
    mse = mean_squared_error(y_test_np, test_outputs_np)
    print(f"Test MSE: {mse:.4f}")


plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
