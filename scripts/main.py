from sklearn.model_selection import train_test_split

from src.normal_method.data import hadamard_total, random_total
from src.normal_method.speckle import inv_hadamard
from src.normal_method.speckle.prediction import (
    Original_pred,
    random_pattern_split,
    speckle_noise_calculation,
)
from src.normal_method.visualization.display import (
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
X_train1, X_train2, y_train1, y_train2 = random_pattern_split()
# y_hadamard.shape: (64, 500)
# X_hadamard.shape: (64, 64)
X_hadamard, y_hadamard = hadamard_total()
print(hadamard_sp.shape)  # 500, 64
print(X_hadamard.shape)  # (64, 64)
print(y_hadamard.shape)  # (64, 500)
print(X_train.shape)  # (450, 64)
S_nn = Original_pred()
display_image_random(2, S_nn, X_train1, y_train1)
display_image_random(2, S_nn, X_train2, y_train2)
# display_image_hadamard(4, hadamard_sp, X_hadamard, y_hadamard)
# display_image_random(2, predicted_speckle, X_train, y_train)
# display_image_random(2, predicted_speckle, X_test, y_test)
