import numpy as np

file_path = "data/raw/HP_mosaic_random_size8x8_image64+10+500_alternate.npz"
collection = "data/raw/HP+mosaic+rand_image64+10+500_size8x8_alternate_200x20020240618_collect.npz"

# Load the data
using_data = np.load(file_path)
collection_data = np.load(collection)
