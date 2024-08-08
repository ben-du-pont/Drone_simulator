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
csv_path = csv_dir / 'outlier_filtering.csv'

# Load the data
data = pd.read_csv(csv_path, header=None)

data.columns = ['number_of_measurements', 'number_of_measurements_outlier_filtering', 'min_angle_between_two_consecutive_measurements', 'min_distance_to_anchor', 'max_distance_to_anchor', 'mean_distance_to_anchor', 'std_distance_to_anchor', 'angular_span_elevation', 'angular_span_azimuth', 'constant_bias_gt', 'linear_bias_gt', 'measured_noise_mean', 'noise_variance', 'measured_noise_var', 'outlier_probability', 'linear_error', 'linear_error_full', 'linear_linear_bias_error', 'linear_constant_bias_error', 'linear_error_outlier_filtering', 'linear_error_full_outlier_filtering', 'linear_linear_bias_error_outlier_filtering', 'linear_constant_bias_error_outlier_filtering', 'error', 'error_full', 'constant_bias_error', 'linear_bias_error', 'error_outlier_filtering', 'error_full_outlier_filtering', 'constant_bias_error_outlier_filtering' ,'linear_bias_error_outlier_filtering']

# Extract relevant columns for linear and non-linear comparisons
linear_comparison_data = data[['linear_error', 'linear_error_full', 'linear_linear_bias_error', 'linear_constant_bias_error',
                               'linear_error_outlier_filtering', 'linear_error_full_outlier_filtering', 'linear_linear_bias_error_outlier_filtering', 'linear_constant_bias_error_outlier_filtering']]

non_linear_comparison_data = data[['error', 'error_full', 'constant_bias_error', 'linear_bias_error',
                                   'error_outlier_filtering', 'error_full_outlier_filtering', 'constant_bias_error_outlier_filtering', 'linear_bias_error_outlier_filtering']]


linear_comparison_data_melted = pd.melt(linear_comparison_data, 
                                        value_vars=['linear_error', 'linear_error_full', 'linear_linear_bias_error', 'linear_constant_bias_error',
                                                    'linear_error_outlier_filtering', 'linear_error_full_outlier_filtering', 'linear_linear_bias_error_outlier_filtering', 'linear_constant_bias_error_outlier_filtering'],
                                        var_name='Metric',
                                        value_name='Error')

non_linear_comparison_data_melted = pd.melt(non_linear_comparison_data, 
                                            value_vars=['error', 'error_full', 'constant_bias_error', 'linear_bias_error',
                                                        'error_outlier_filtering', 'error_full_outlier_filtering', 'constant_bias_error_outlier_filtering', 'linear_bias_error_outlier_filtering'],
                                            var_name='Metric',
                                            value_name='Error')

# Define metric labels
linear_metric_labels = {
    'linear_error': 'Linear Error',
    'linear_error_full': 'Linear Full Error',
    'linear_linear_bias_error': 'Linear Linear Bias Error',
    'linear_constant_bias_error': 'Linear Constant Bias Error',
    'linear_error_outlier_filtering': 'Linear Error (Filtered)',
    'linear_error_full_outlier_filtering': 'Linear Full Error (Filtered)',
    'linear_linear_bias_error_outlier_filtering': 'Linear Linear Bias Error (Filtered)',
    'linear_constant_bias_error_outlier_filtering': 'Linear Constant Bias Error (Filtered)'
}

non_linear_metric_labels = {
    'error': 'Non-Linear Error',
    'error_full': 'Non-Linear Full Error',
    'constant_bias_error': 'Non-Linear Constant Bias Error',
    'linear_bias_error': 'Non-Linear Linear Bias Error',
    'error_outlier_filtering': 'Non-Linear Error (Filtered)',
    'error_full_outlier_filtering': 'Non-Linear Full Error (Filtered)',
    'constant_bias_error_outlier_filtering': 'Non-Linear Constant Bias Error (Filtered)',
    'linear_bias_error_outlier_filtering': 'Non-Linear Linear Bias Error (Filtered)'
}

linear_comparison_data_melted['Metric'] = linear_comparison_data_melted['Metric'].map(linear_metric_labels)
non_linear_comparison_data_melted['Metric'] = non_linear_comparison_data_melted['Metric'].map(non_linear_metric_labels)

# Plot Linear Comparison
plt.figure(figsize=(16, 10))

# Linear Metrics
plt.subplot(2, 1, 1)
sns.boxplot(x='Metric', y='Error', data=linear_comparison_data_melted, palette='Set2')
plt.title('Comparison of Linear Estimation Precision: With and Without Outlier Filtering')
plt.xticks(rotation=45, ha='right')

# Non-Linear Metrics
plt.subplot(2, 1, 2)
sns.boxplot(x='Metric', y='Error', data=non_linear_comparison_data_melted, palette='Set1')
plt.title('Comparison of Non-Linear Estimation Precision: With and Without Outlier Filtering')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()