import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.normal_method.data import random_total
from src.normal_method.speckle.prediction import inv_hadamard

X_random, y_random = random_total()
X_train, X_test, y_train, y_test = train_test_split(
    X_random, y_random, test_size=0.1, shuffle=False
)
S_hd = inv_hadamard()

delta = y_train - np.dot(X_train, S_hd.T)
delta_ridge = Ridge(alpha=5000, solver="sparse_cg", max_iter=10000000)
delta_ridge.fit(X_train, delta)
delta_ridge_coef = delta_ridge.coef_
pred_S = S_hd + delta_ridge_coef


def inverse_mat(j, sp, target_train, y_train, target_test, y_test):
    aa = np.linalg.pinv(sp)
    bb = np.dot(aa, y_train.T).T
    bb_test = np.dot(aa, y_test.T).T
    fig = plt.figure(figsize=(10, 4 * j))
    for i in range(j):
        ax1 = fig.add_subplot(j, 2, i * 2 + 1)
        ax2 = fig.add_subplot(j, 2, i * 2 + 2)

        ax1.set_title("Target_image")
        ax2.set_title("Reconstruction")

        ax1.imshow(target_train[i, :].reshape(8, 8), cmap="gray", vmin=-1, vmax=1)
        ax2.imshow(bb[i, :].reshape(8, 8), cmap="gray")

    print(
        "Train MSE =",
        mean_squared_error(target_train, bb),
        "\n",
        "Test MSE =",
        mean_squared_error(target_test, bb_test),
    )


def inverse_mat_arr(sp, target, yy):
    aa = np.dot(sp, target.T).T
    print("MSE =", mean_squared_error(yy, aa))


if __name__ == "__main__":
    inverse_mat(4, pred_S, X_train, y_train, X_test, y_test)
    # inverse_mat_arr(pred_S, X_test, y_test)
    # plt.tight_layout()
    plt.show()
