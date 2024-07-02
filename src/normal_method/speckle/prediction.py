import numpy as np

from src.normal_method.data import hadamard_total


def inv_hadamard():
    X_hadamard, y_hadamard = hadamard_total()
    n = int(X_hadamard.shape[0])
    hadamard_sp = np.dot(y_hadamard.T, X_hadamard.T) / n
    # hadamard_sp.shape: (500, 64)
    return hadamard_sp
