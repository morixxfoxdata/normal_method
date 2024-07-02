from src.normal_method.data import hadamard_total
from src.normal_method.speckle import inv_hadamard, speckle_noise_calculation
from src.normal_method.visualization.display import display_image

hadamard_sp = inv_hadamard()
predicted_speckle = speckle_noise_calculation(hadamard_sp)
X_hadamard, y_hadamard = hadamard_total()

display_image(2, X_hadamard, predicted_speckle)
