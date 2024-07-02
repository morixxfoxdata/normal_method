import matplotlib.pyplot as plt

from src.normal_method.data import hadamard_total
from src.normal_method.speckle import inv_hadamard, speckle_noise_calculation


def display_image(j, xx, yy, size=8):
    fig = plt.figure(figsize=(10, 4 * j))
    for i in range(j):
        ax1 = fig.add_subplot(j, 2, i * 2 + 1)
        ax2 = fig.add_subplot(j, 2, i * 2 + 2)

        ax1.set_title("Target_image")
        ax2.set_title("Reconstruction")

        ax1.imshow(xx[i, :].reshape(size, size), cmap="gray", vmin=-1, vmax=1)
        ax2.imshow(yy[i, :].reshape(size, size), cmap="gray")
    # print("MSE =", mean_squared_error(xx, yy))
    plt.show()


hadamard_sp = inv_hadamard()
predicted_speckle = speckle_noise_calculation(hadamard_sp)
X_hadamard, y_hadamard = hadamard_total()

# display_image(2, X_hadamard, predicted_speckle)
# print(predicted_speckle.shape)
