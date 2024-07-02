import numpy as np
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
    X_train1 = X_train[0 : int(X_train.shape[0] / 2), :]
    X_train2 = X_train[int(X_train.shape[0] / 2) :, :]

    y_train1 = y_train[0 : int(X_train.shape[0] / 2), :]
    y_train2 = y_train[int(X_train.shape[0] / 2) :, :]
    return X_train1, X_train2, y_train1, y_train2


def speckle_noise_calculation(S):
    _, X_train2, _, y_train2 = random_pattern_split()
    delta = y_train2 - np.dot(X_train2, S.T)
    delta_Ridge = Ridge(alpha=0.001)
    delta_Ridge.fit(X_train2, delta)
    delta_ridge_coef = delta_Ridge.coef_
    predicted_speckle = S + delta_ridge_coef
    # speckle.shape: (500, 64)
    return predicted_speckle
