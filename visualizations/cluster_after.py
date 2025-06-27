import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from sklearn.datasets import make_blobs


# Set random seed for reproducibility
np.random.seed(42)

# Create figure and axis
plt.figure(figsize=(10, 10))
ax = plt.gca()

# Generate two clusters with MODIFIED sizes
# First cluster (now compressed, but still more points)
X1, _ = make_blobs(n_samples=15, centers=[[5, 5]], cluster_std=0.8, random_state=42)  # Reduced std

# Second cluster (now expanded, but still fewer points)
X2, _ = make_blobs(n_samples=8, centers=[[2, 2]], cluster_std=1.5, random_state=24)  # Increased std

# Plot scatter points
ax.scatter(X1[:, 0], X1[:, 1], color="#d916a8", s=120, alpha=0.9, zorder=3)
ax.scatter(X2[:, 0], X2[:, 1], color="#eebe00", s=120, alpha=0.9, zorder=3)

# Add cluster circles
# Calculate cluster centers
center1 = np.mean(X1, axis=0)
center2 = np.mean(X2, axis=0)

# Both clusters now have the same circle size
radius1 = 1.2  # Same size for both clusters
radius2 = 1.2

# Add inner circles
inner_circle1 = Circle(center1, radius1, fill=True, color="#ff80df", alpha=0.5, zorder=1)
inner_circle2 = Circle(center2, radius2, fill=True, color="#fcce14", alpha=0.5, zorder=1)
ax.add_patch(inner_circle1)
ax.add_patch(inner_circle2)

# Add outer circles with lower opacity
outer_circle1 = Circle(center1, radius1 * 2, fill=True, color="#ff80df", alpha=0.2, zorder=0)
outer_circle2 = Circle(center2, radius2 * 2, fill=True, color="#fcce14", alpha=0.2, zorder=0)
ax.add_patch(outer_circle1)
ax.add_patch(outer_circle2)

# Set background color to white with 50% transparency
ax.set_facecolor("white")
ax.patch.set_alpha(0.5)

# Remove grid
ax.grid(False)

# Set equal aspect ratio
ax.set_aspect("equal")

# Set limits
plt.xlim(0, 8)
plt.ylim(0, 8)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
plt.axis("off")

# Display the plot
plt.tight_layout()
plt.savefig("cluster_visualization_equal_size.png", dpi=300, bbox_inches="tight", transparent=True)
plt.show()
