import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


def display_image_hadamard(j, sp, xx, yy, size=8):
    aa = np.linalg.pinv(sp)
    bb = np.dot(aa, yy.T)
    fig = plt.figure(figsize=(8, 4 * j))
    for i in range(j):
        ax1 = fig.add_subplot(j, 2, i * 2 + 1)
        ax2 = fig.add_subplot(j, 2, i * 2 + 2)

        ax1.set_title("Target_image")
        ax2.set_title("Reconstruction")

        ax1.imshow(xx[i, :].reshape(size, size), cmap="gray", vmin=-1, vmax=1)
        ax2.imshow(bb[i, :].reshape(size, size), cmap="gray")
    print("MSE =", mean_squared_error(xx, bb))
    plt.show()


def display_image_random(j, sp, xx, yy, size=8):
    aa = np.linalg.pinv(sp)
    bb = np.dot(yy, aa.T)
    fig = plt.figure(figsize=(8, 4 * j))
    for i in range(j):
        ax1 = fig.add_subplot(j, 2, i * 2 + 1)
        ax2 = fig.add_subplot(j, 2, i * 2 + 2)

        ax1.set_title("Target_image")
        ax2.set_title("Reconstruction")

        ax1.imshow(xx[i, :].reshape(size, size), cmap="gray", vmin=-1, vmax=1)
        ax2.imshow(bb[i, :].reshape(size, size), cmap="gray")
    print("MSE =", mean_squared_error(xx, bb))
    plt.show()


def image_display(j, xx, yy, size=8):
    print("MSE =", mean_squared_error(xx, yy))
    fig = plt.figure(figsize=(4, 4 * j))
    for i in range(j):
        ax1 = fig.add_subplot(j, 2, i * 2 + 1)
        ax2 = fig.add_subplot(j, 2, i * 2 + 2)

        ax1.set_title("Target_image")
        ax2.set_title("Reconstruction")

        ax1.imshow(xx[i, :].reshape(size, size), cmap="gray", vmin=-1, vmax=1)
        ax2.imshow(yy[i, :].reshape(size, size), cmap="gray")
    plt.show()
