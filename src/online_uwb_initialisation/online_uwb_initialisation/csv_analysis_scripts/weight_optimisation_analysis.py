from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json 
import seaborn as sns

import numpy as np
package_path = Path(__file__).parent.resolve()
csv_dir = package_path

csv_path = csv_dir / 'error_values.csv'



# Read the CSV file
df_environments = pd.read_csv(csv_dir / 'sim_environments.csv', header=None)


# Deserialize JSON strings
for col in df_environments.columns:
    df_environments[col] = df_environments[col].apply(lambda x: json.loads(x) if isinstance(x, str) and (x.startswith('[') or x.startswith('{')) else x)

# Loop through each row (environment) in the DataFrame
for index, row in df_environments.iterrows():
    # Extract data from DataFrame
    environment = row[0]
    trajectory = row[1]
    waypoints = row[2]
    base_anchors = row[3]
    unknown_anchors = row[4]

    # Convert extracted data to numpy arrays for plotting
    trajectory_x, trajectory_y, trajectory_z = map(np.array, trajectory)
    waypoints = np.array(waypoints)
    base_anchors_coords = np.array([anchor[0] for anchor in base_anchors])
    unknown_anchors_coords = np.array([anchor[0] for anchor in unknown_anchors])

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot waypoints
    waypoints_x, waypoints_y, waypoints_z = waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]
    ax.scatter(waypoints_x, waypoints_y, waypoints_z, color='blue', label='Waypoints')

    # Plot trajectory
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='green', label='Trajectory')

    # Plot base anchors
    base_anchors_x, base_anchors_y, base_anchors_z = base_anchors_coords[:, 0], base_anchors_coords[:, 1], base_anchors_coords[:, 2]
    ax.scatter(base_anchors_x, base_anchors_y, base_anchors_z, color='red', marker='x', label='Base Anchors')

    # Plot unknown anchors
    unknown_anchors_x, unknown_anchors_y, unknown_anchors_z = unknown_anchors_coords[:, 0], unknown_anchors_coords[:, 1], unknown_anchors_coords[:, 2]
    ax.scatter(unknown_anchors_x, unknown_anchors_y, unknown_anchors_z, color='purple', marker='x', label='Unknown Anchors')

    # Set plot labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'Waypoints, Trajectory, and Anchors in 3D for Environment {index}')
    ax.legend()

    # Show plot
    plt.show()














df = pd.read_csv(csv_path)


print(df.keys())

# Scatter plots for each hyperparameter
hyperparameters = ['distance_to_anchor_threshold', 'gdop_threshold', 'weight_angle', 'weight_distance', 'weight_dev', 'num_measurements']
for hyperparameter in hyperparameters:
    plt.figure()
    plt.xlabel(hyperparameter)
    plt.ylabel('Error')
    plt.title(f'Error vs {hyperparameter}')
    plt.grid(True)
    
    # Remove outliers
    q1 = df['final_estimate_error'].quantile(0.25)
    q3 = df['final_estimate_error'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_df = df[(df['final_estimate_error'] >= lower_bound) & (df['final_estimate_error'] <= upper_bound)]
    
    sns.scatterplot(data=filtered_df, x=hyperparameter, y='final_estimate_error', color='red')
    plt.xscale('log')  # Set x-axis to log scale
    plt.show()

# # Pairplot to visualize the relationships between hyperparameters
# sns.pairplot(df[hyperparameters + ['final_estimate_error']])
# plt.show()

# Heatmap to visualize the correlation between hyperparameters and error
plt.figure()
sns.heatmap(df[hyperparameters + ['final_estimate_error']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Box plots to compare the distribution of error for each hyperparameter
for hyperparameter in hyperparameters:
    plt.figure()
    sns.boxplot(data=filtered_df, x=hyperparameter, y='final_estimate_error')
    plt.title(f'Error Distribution for {hyperparameter}')
    plt.show()

# Violin plots to compare the distribution of error for each hyperparameter
for hyperparameter in hyperparameters:
    plt.figure()
    sns.violinplot(data=filtered_df, x=hyperparameter, y='final_estimate_error')
    plt.title(f'Error Distribution for {hyperparameter}')
    plt.show()


average_error = df['final_estimate_error'].mean()
print(f"Average Final Estimator Error: {average_error}")

average_error = df['final_estimate_error'].median()
print(f"Median Final Estimator Error: {average_error}")