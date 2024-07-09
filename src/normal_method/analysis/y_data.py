import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Perform principal component analysis (PCA)
from sklearn.decomposition import PCA

from src.normal_method.data import mnist_total

XX, yy = mnist_total()


def compute_statistics(data):
    """Compute various statistics for the given data."""
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    return mean, median, std_dev, skewness, kurtosis


# Compute statistics for each image
stats_data = np.array([compute_statistics(img) for img in yy])

# Create labels for each statistic
stat_labels = ["Mean", "Median", "Std Dev", "Skewness", "Kurtosis"]

# Plot the statistics
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Statistical Comparison of Speckle Patterns", fontsize=16)
for i, label in enumerate(stat_labels):
    ax = axs[i // 3, i % 3] if i < 5 else axs[1, 2]
    ax.boxplot(stats_data[:, i])
    ax.set_title(label)
    ax.set_xticklabels([])

# Remove the empty subplot
axs[1, 2].axis("off")

plt.tight_layout()
plt.show()

# Print summary statistics
print("Summary Statistics:")
for i, label in enumerate(stat_labels):
    print(f"{label}:")
    print(f"  Min: {np.min(stats_data[:, i]):.4f}")
    print(f"  Max: {np.max(stats_data[:, i]):.4f}")
    print(f"  Mean: {np.mean(stats_data[:, i]):.4f}")
    print(f"  Std Dev: {np.std(stats_data[:, i]):.4f}")
    print()

# Compute correlation matrix
corr_matrix = np.corrcoef(yy)

# Plot correlation matrix heatmap
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.title("Correlation Matrix of Speckle Patterns")
plt.xlabel("Image Index")
plt.ylabel("Image Index")
plt.show()


pca = PCA()
pca.fit(yy)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("PCA: Cumulative Explained Variance Ratio")
plt.grid(True)
plt.show()

print(
    f"Number of components to explain 95% of variance: {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1}"
)
