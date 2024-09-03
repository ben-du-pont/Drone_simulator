import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

from pathlib import Path

package_path = Path(__file__).parent.parent.resolve()
csv_dir = package_path / 'csv_files'
csv_path = csv_dir / 'trajectory_optimisation.csv'

# Load the data
data = pd.read_csv(csv_path, header=None)

data.columns = ['number_of_measurements', 'min_angle_between_two_consecutive_measurements', 'min_distance_to_anchor', 'max_distance_to_anchor', 'mean_distance_to_anchor', 'std_distance_to_anchor', 'angular_span_elevation', 'angular_span_azimuth', 'constant_bias_gt', 'linear_bias_gt', 'measured_noise_mean', 'noise_variance', 'measured_noise_var', 'outlier_probability', 'linear_error', 'linear_error_full', 'linear_linear_bias_error', 'linear_constant_bias_error', 'error_non_linear', 'error_non_linear_full', 'constant_bias_error_non_linear', 'linear_bias_error_non_linear', 'error_final', 'error_final_full', 'num_optimal_waypoints']

# Define configurations
configs = ['linear_error', 'error_non_linear', 'error_final']
config_labels = ['Linear Error', 'Refined non linear error', 'Final error after optimal waypoints']

# Create a DataFrame for plotting
plot_data = pd.melt(data[configs], var_name='Configuration', value_name='Position Error')
plot_data['Configuration'] = plot_data['Configuration'].map(dict(zip(configs, config_labels)))

# Plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Configuration', y='Position Error', data=plot_data)
plt.title('Position Error Comparison for Nonlinear Optimization Methods')
plt.xticks(rotation=45)
plt.show()