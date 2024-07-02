from src.normal_method.data import hadamard_total
from src.normal_method.speckle import inv_hadamard
from src.normal_method.visualization.display import display_image

hadamard_sp = inv_hadamard()
# predicted_speckle = speckle_noise_calculation(hadamard_sp)
X_hadamard, y_hadamard = hadamard_total()
print(hadamard_sp.shape)
print(X_hadamard.shape)
print(y_hadamard.shape)
display_image(4, hadamard_sp, X_hadamard, y_hadamard)
