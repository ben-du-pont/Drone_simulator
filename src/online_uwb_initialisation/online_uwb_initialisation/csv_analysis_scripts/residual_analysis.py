import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
# Ground truth position
ground_truth = np.array([10, 20, 30])

# Measurement positions (x, y, z)
measurements = np.array([
    [9.5, 21, 29],
    [10.2, 19.8, 30.5],
    [11, 20.1, 29.9],
    [9.8, 20.3, 30.2]
])

# Residuals (magnitude of error in the estimation)
residuals = np.array([1.0, 0.7, 0.5, 1.2])

# Normalize residuals for color mapping
norm = plt.Normalize(residuals.min(), residuals.max())

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot ground truth position
ax.scatter(ground_truth[0], ground_truth[1], ground_truth[2], c='r', marker='o', s=100, label='Ground Truth')

# Plot measurement positions with color-coded residuals
scatter = ax.scatter(measurements[:, 0], measurements[:, 1], measurements[:, 2], 
                     c=residuals, cmap='coolwarm', norm=norm, s=50)  # Changed colormap to coolwarm

# Add color bar to show the scale of residuals
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Residual Magnitude')

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot with Ground Truth and Measurements')

# Remove the legend entry for 'Measurements'
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=[handles[0]], labels=[labels[0]])  # Only keep the legend for 'Ground Truth'

# Show plot
plt.show()
