import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from pathlib import Path

# Load data
package_path = Path(__file__).parent.resolve()
csv_dir = package_path / 'csv_files/insane_dataset'

uwb_data = pd.read_csv(csv_dir / 'pick_and_place/uwb_range.csv')
mocap_data = pd.read_csv(csv_dir / 'pick_and_place/mocap_vehicle_data.csv')


valid_indices_m1 = uwb_data['valid_m1'] == 1
dist_m1 = uwb_data.loc[valid_indices_m1, 'dist_m1'].values
timestamps_m1 = uwb_data.loc[valid_indices_m1, 't'].values

valid_indices_m2 = uwb_data['valid_m2'] == 1
dist_m2 = uwb_data.loc[valid_indices_m2, 'dist_m2'].values
timestamps_m2 = uwb_data.loc[valid_indices_m2, 't'].values

valid_indices_m3 = uwb_data['valid_m3'] == 1
dist_m3 = uwb_data.loc[valid_indices_m3, 'dist_m3'].values
timestamps_m3 = uwb_data.loc[valid_indices_m3, 't'].values

# Interpolate motion capture data
mocap_interp_x = interp1d(mocap_data['t'], mocap_data['p_x'], kind='linear', fill_value="extrapolate")
mocap_interp_y = interp1d(mocap_data['t'], mocap_data['p_y'], kind='linear', fill_value="extrapolate")
mocap_interp_z = interp1d(mocap_data['t'], mocap_data['p_z'], kind='linear', fill_value="extrapolate")

# Function to estimate anchor position
def estimate_anchor_position(distances, timestamps):
    # Get interpolated positions
    positions_x = mocap_interp_x(timestamps)
    positions_y = mocap_interp_y(timestamps)
    positions_z = mocap_interp_z(timestamps)
    positions = np.vstack((positions_x, positions_y, positions_z)).T

    # Objective function to minimize
    def objective(anchor_pos):
        distances_calc = np.sqrt(np.sum((positions - anchor_pos)**2, axis=1))
        return np.sum((distances_calc - distances)**2)

    # Initial guess for anchor position (e.g., origin)
    initial_guess = np.array([0.0, 0.0, 0.0])

    # Minimize the objective function
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    return result.x

# Estimate anchor positions
anchor_position_m1 = estimate_anchor_position(dist_m1, timestamps_m1)
anchor_position_m2 = estimate_anchor_position(dist_m2, timestamps_m2)
anchor_position_m3 = estimate_anchor_position(dist_m3, timestamps_m3)

print("Estimated position of anchor 1:", anchor_position_m1)
print("Estimated position of anchor 2:", anchor_position_m2)
print("Estimated position of anchor 3:", anchor_position_m3)

# Calculate distance errors and true distances
def calculate_distance_errors_and_true_distances(anchor_position, distances, timestamps):
    positions_x = mocap_interp_x(timestamps)
    positions_y = mocap_interp_y(timestamps)
    positions_z = mocap_interp_z(timestamps)
    positions = np.vstack((positions_x, positions_y, positions_z)).T
    true_distances = np.sqrt(np.sum((positions - anchor_position)**2, axis=1))
    errors = distances - true_distances
    return errors, true_distances

distance_errors_m1, true_distances_m1 = calculate_distance_errors_and_true_distances(anchor_position_m1, dist_m1, timestamps_m1)
distance_errors_m2, true_distances_m2 = calculate_distance_errors_and_true_distances(anchor_position_m2, dist_m2, timestamps_m2)
distance_errors_m3, true_distances_m3 = calculate_distance_errors_and_true_distances(anchor_position_m3, dist_m3, timestamps_m3)


# Get interpolated positions for the entire motion capture dataset
mocap_positions_x = mocap_data['p_x'].values
mocap_positions_y = mocap_data['p_y'].values
mocap_positions_z = mocap_data['p_z'].values

# Plotting the drone trajectory and anchor positions
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the drone trajectory
ax.plot(mocap_positions_x, mocap_positions_y, mocap_positions_z, label='Drone Trajectory', color='blue')

# Plot the anchors
ax.scatter(*anchor_position_m1, color='red', s=100, label='Anchor 1')
ax.scatter(*anchor_position_m2, color='green', s=100, label='Anchor 2')
ax.scatter(*anchor_position_m3, color='orange', s=100, label='Anchor 3')

# Plot starting and ending points
start_point = (mocap_positions_x[0], mocap_positions_y[0], mocap_positions_z[0])
end_point = (mocap_positions_x[-1], mocap_positions_y[-1], mocap_positions_z[-1])


ax.scatter(*start_point, color='black', s=300, edgecolor='white', label='Start Point', marker='o')  # Large black circle
ax.scatter(*end_point, color='black', s=300, edgecolor='white', label='End Point', marker='x')    # Large black cross


# Labels and legend
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Drone Trajectory and Anchor Positions')
ax.legend()

plt.show()



# Plot the distance errors and true distances over time for each anchor with secondary y-axis for errors
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(timestamps_m1, true_distances_m1, marker='x', linestyle='--', color='b', label='True Distance')
plt.xlabel('Time (s)')
plt.ylabel('True Distance (m)')
plt.title('Distance Error and True Distance Over Time for Anchor 1')
plt.legend(loc='upper left')
plt.grid(True)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(timestamps_m1, distance_errors_m1, marker='o', linestyle='-', color='r', label='Distance Error')
ax2.set_ylabel('Distance Error (m)')
ax2.legend(loc='upper right')

plt.subplot(3, 1, 2)
plt.plot(timestamps_m2, true_distances_m2, marker='x', linestyle='--', color='b', label='True Distance')
plt.xlabel('Time (s)')
plt.ylabel('True Distance (m)')
plt.title('Distance Error and True Distance Over Time for Anchor 2')
plt.legend(loc='upper left')
plt.grid(True)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(timestamps_m2, distance_errors_m2, marker='o', linestyle='-', color='r', label='Distance Error')
ax2.set_ylabel('Distance Error (m)')
ax2.legend(loc='upper right')

plt.subplot(3, 1, 3)
plt.plot(timestamps_m3, true_distances_m3, marker='x', linestyle='--', color='b', label='True Distance')
plt.xlabel('Time (s)')
plt.ylabel('True Distance (m)')
plt.title('Distance Error and True Distance Over Time for Anchor 3')
plt.legend(loc='upper left')
plt.grid(True)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(timestamps_m3, distance_errors_m3, marker='o', linestyle='-', color='r', label='Distance Error')
ax2.set_ylabel('Distance Error (m)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()







































# Load data
uwb_data = pd.read_csv(csv_dir / 'open_field/uwb_range.csv')
ground_truth_data = pd.read_csv(csv_dir / 'open_field/ground_truth_80hz.csv')

valid_indices_m1 = uwb_data['valid_m1'] == 1
dist_m1 = uwb_data.loc[valid_indices_m1, 'dist_m1'].values
timestamps_m1 = uwb_data.loc[valid_indices_m1, 't'].values

valid_indices_m2 = uwb_data['valid_m2'] == 1
dist_m2 = uwb_data.loc[valid_indices_m2, 'dist_m2'].values
timestamps_m2 = uwb_data.loc[valid_indices_m2, 't'].values

valid_indices_m3 = uwb_data['valid_m3'] == 1
dist_m3 = uwb_data.loc[valid_indices_m3, 'dist_m3'].values
timestamps_m3 = uwb_data.loc[valid_indices_m3, 't'].values

# Interpolate motion capture data
ground_truth_interp_x = interp1d(ground_truth_data['t'], ground_truth_data['p_x'], kind='linear', fill_value="extrapolate")
ground_truth_interp_y = interp1d(ground_truth_data['t'], ground_truth_data['p_y'], kind='linear', fill_value="extrapolate")
ground_truth_interp_z = interp1d(ground_truth_data['t'], ground_truth_data['p_z'], kind='linear', fill_value="extrapolate")

# Function to estimate anchor position
def estimate_anchor_position(distances, timestamps):
    # Get interpolated positions
    positions_x = ground_truth_interp_x(timestamps)
    positions_y = ground_truth_interp_y(timestamps)
    positions_z = ground_truth_interp_z(timestamps)
    positions = np.vstack((positions_x, positions_y, positions_z)).T

    # Objective function to minimize
    def objective(anchor_pos):
        distances_calc = np.sqrt(np.sum((positions - anchor_pos)**2, axis=1))
        return np.sum((distances_calc - distances)**2)

    # Initial guess for anchor position (e.g., origin)
    initial_guess = np.array([0.0, 0.0, 0.0])

    # Minimize the objective function
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    return result.x

# Estimate anchor positions
anchor_position_m1 = estimate_anchor_position(dist_m1, timestamps_m1)
anchor_position_m2 = estimate_anchor_position(dist_m2, timestamps_m2)
anchor_position_m3 = estimate_anchor_position(dist_m3, timestamps_m3)

print("Estimated position of anchor 1:", anchor_position_m1)
print("Estimated position of anchor 2:", anchor_position_m2)
print("Estimated position of anchor 3:", anchor_position_m3)

# Calculate distance errors and true distances
def calculate_distance_errors_and_true_distances(anchor_position, distances, timestamps):
    positions_x = ground_truth_interp_x(timestamps)
    positions_y = ground_truth_interp_y(timestamps)
    positions_z = ground_truth_interp_z(timestamps)
    positions = np.vstack((positions_x, positions_y, positions_z)).T
    true_distances = np.sqrt(np.sum((positions - anchor_position)**2, axis=1))
    errors = distances - true_distances
    return errors, true_distances

distance_errors_m1, true_distances_m1 = calculate_distance_errors_and_true_distances(anchor_position_m1, dist_m1, timestamps_m1)
distance_errors_m2, true_distances_m2 = calculate_distance_errors_and_true_distances(anchor_position_m2, dist_m2, timestamps_m2)
distance_errors_m3, true_distances_m3 = calculate_distance_errors_and_true_distances(anchor_position_m3, dist_m3, timestamps_m3)


# Get interpolated positions for the entire motion capture dataset
ground_truth_positions_x = ground_truth_data['p_x'].values
ground_truth_positions_y = ground_truth_data['p_y'].values
ground_truth_positions_z = ground_truth_data['p_z'].values

# Plotting the drone trajectory and anchor positions
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the drone trajectory
ax.plot(ground_truth_positions_x, ground_truth_positions_y, ground_truth_positions_z, label='Drone Trajectory', color='blue')

# Plot the anchors
ax.scatter(*anchor_position_m1, color='red', s=100, label='Anchor 1')
ax.scatter(*anchor_position_m2, color='green', s=100, label='Anchor 2')
ax.scatter(*anchor_position_m3, color='orange', s=100, label='Anchor 3')

# Plot starting and ending points
start_point = (ground_truth_positions_x[0], ground_truth_positions_y[0], ground_truth_positions_z[0])
end_point = (ground_truth_positions_x[-1], ground_truth_positions_y[-1], ground_truth_positions_z[-1])


ax.scatter(*start_point, color='black', s=300, edgecolor='white', label='Start Point', marker='o')  # Large black circle
ax.scatter(*end_point, color='black', s=300, edgecolor='white', label='End Point', marker='x')    # Large black cross


# Labels and legend
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Drone Trajectory and Anchor Positions')
ax.legend()

plt.show()



# Plot the distance errors and true distances over time for each anchor with secondary y-axis for errors
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(timestamps_m1, true_distances_m1, marker='x', linestyle='--', color='b', label='True Distance')
plt.xlabel('Time (s)')
plt.ylabel('True Distance (m)')
plt.title('Distance Error and True Distance Over Time for Anchor 1')
plt.legend(loc='upper left')
plt.grid(True)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(timestamps_m1, distance_errors_m1, marker='o', linestyle='-', color='r', label='Distance Error')
ax2.set_ylabel('Distance Error (m)')
ax2.legend(loc='upper right')

plt.subplot(3, 1, 2)
plt.plot(timestamps_m2, true_distances_m2, marker='x', linestyle='--', color='b', label='True Distance')
plt.xlabel('Time (s)')
plt.ylabel('True Distance (m)')
plt.title('Distance Error and True Distance Over Time for Anchor 2')
plt.legend(loc='upper left')
plt.grid(True)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(timestamps_m2, distance_errors_m2, marker='o', linestyle='-', color='r', label='Distance Error')
ax2.set_ylabel('Distance Error (m)')
ax2.legend(loc='upper right')

plt.subplot(3, 1, 3)
plt.plot(timestamps_m3, true_distances_m3, marker='x', linestyle='--', color='b', label='True Distance')
plt.xlabel('Time (s)')
plt.ylabel('True Distance (m)')
plt.title('Distance Error and True Distance Over Time for Anchor 3')
plt.legend(loc='upper left')
plt.grid(True)

ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.plot(timestamps_m3, distance_errors_m3, marker='o', linestyle='-', color='r', label='Distance Error')
ax2.set_ylabel('Distance Error (m)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()


