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
csv_path = csv_dir / 'clustering.csv'

# Load the data
data = pd.read_csv(csv_path, header=None)

data.columns = ['number_of_measurements', 'min_angle_between_two_consecutive_measurements', 'min_distance_to_anchor', 'max_distance_to_anchor', 'mean_distance_to_anchor', 'std_distance_to_anchor', 'angular_span_elevation', 'angular_span_azimuth', 'constant_bias_gt', 'linear_bias_gt', 'measured_noise_mean', 'noise_variance', 'measured_noise_var', 'outlier_probability', 'linear_error', 'linear_error_full', 'linear_linear_bias_error', 'linear_constant_bias_error', 'error', 'error_full', 'constant_bias_error', 'linear_bias_error', 'error_cluster', 'error_cluster_full', 'constant_bias_error_cluster', 'linear_bias_error_cluster']

# Extract relevant columns for non-clustered and clustered comparison
comparison_data = data[['error', 'error_full', 'constant_bias_error', 'linear_bias_error',
                        'error_cluster', 'error_cluster_full', 'constant_bias_error_cluster', 'linear_bias_error_cluster']]

# Melt the DataFrame for easier plotting
comparison_data_melted = pd.melt(comparison_data, 
                                  value_vars=['error', 'error_full', 'constant_bias_error', 'linear_bias_error',
                                              'error_cluster', 'error_cluster_full', 'constant_bias_error_cluster', 'linear_bias_error_cluster'],
                                  var_name='Metric',
                                  value_name='Error')

# Define metric labels
metric_labels = {
    'error': 'Non-Clustered Position Error',
    'error_full': 'Non-Clustered Full Error',
    'constant_bias_error': 'Non-Clustered Constant Bias Error',
    'linear_bias_error': 'Non-Clustered Linear Bias Error',
    'error_cluster': 'Clustered Position Error',
    'error_cluster_full': 'Clustered Full Error',
    'constant_bias_error_cluster': 'Clustered Constant Bias Error',
    'linear_bias_error_cluster': 'Clustered Linear Bias Error'
}

comparison_data_melted['Metric'] = comparison_data_melted['Metric'].map(metric_labels)

# Plot the comparison
plt.figure(figsize=(14, 8))
sns.boxplot(x='Metric', y='Error', data=comparison_data_melted, palette='Set2')
plt.title('Comparison of Estimation Precision: Non-Clustered vs. Clustered Measurements')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()