from sklearn.model_selection import train_test_split

from src.normal_method.data import hadamard_total, random_total
from src.normal_method.speckle import inv_hadamard
from src.normal_method.speckle.prediction import speckle_noise_calculation
from src.normal_method.visualization.display import (
    display_image_hadamard,
    display_image_random,
)

# hadamard_sp: (500, 64)
hadamard_sp = inv_hadamard()
predicted_speckle = speckle_noise_calculation(hadamard_sp)
# y_random.shape: (500, 500)
# X_random.shape: (500, 64)
X_random, y_random = random_total()
X_train, X_test, y_train, y_test = train_test_split(
    X_random, y_random, test_size=0.1, shuffle=False
)
# y_hadamard.shape: (64, 500)
# X_hadamard.shape: (64, 64)
X_hadamard, y_hadamard = hadamard_total()
print(hadamard_sp.shape)
print(X_hadamard.shape)
print(y_hadamard.shape)
display_image_hadamard(4, hadamard_sp, X_hadamard, y_hadamard)
display_image_random(2, predicted_speckle, X_train, y_train)
display_image_random(2, predicted_speckle, X_test, y_test)
