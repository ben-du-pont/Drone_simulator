import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from pathlib import Path

# Load the CSV data
package_path = Path(__file__).parent.resolve()
csv_path = package_path / 'error_values.csv'
data = pd.read_csv(csv_path)

# Helper function to extract x, y, z components from the estimator
def extract_components(series, component_indices):
    return series.apply(lambda x: np.array([float(num) for num in re.split('[, ]+', x.replace('(', '').replace(')', '')) if num])[component_indices])

# Define component indices
position_indices = [0, 1, 2]
full_indices = [0, 1, 2, 3, 4]

# Extract components for the estimators and ground truth anchors
data['final_estimator_components'] = extract_components(data['final_estimator'], full_indices)
data['original_estimator_components'] = extract_components(data['original_estimator'], full_indices)
data['gt_anchor_components'] = extract_components(data['gt_anchor'], full_indices)

# Extract components for the estimators and ground truth anchors
data['linear_estimator_components'] = extract_components(data['rough_estimator_linear'], full_indices)
data['gt_anchor_components'] = extract_components(data['gt_anchor'], full_indices)

# Calculate position and full anchor errors
data['position_error'] = data.apply(lambda row: np.linalg.norm(row['final_estimator_components'][:3] - row['gt_anchor_components'][:3]), axis=1)
data['full_anchor_error'] = data.apply(lambda row: np.linalg.norm(row['final_estimator_components'] - row['gt_anchor_components']), axis=1)

# Calculate position and full anchor errors
data['position_error_linear'] = data.apply(lambda row: np.linalg.norm(row['linear_estimator_components'][:3] - row['gt_anchor_components'][:3]), axis=1)
data['full_anchor_error_linear'] = data.apply(lambda row: np.linalg.norm(row['linear_estimator_components'] - row['gt_anchor_components']), axis=1)

# Function to filter outliers based on IQR
def remove_outliers(df, columns):
    filtered_data = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_data = filtered_data[(filtered_data[col] >= lower_bound) & (filtered_data[col] <= upper_bound)]
    return filtered_data

# Filter outliers from 'position_error' and 'full_anchor_error'
filtered_data = remove_outliers(data, ['position_error', 'full_anchor_error'])

# Visualize the effects of biases (True-True, True-False, False-False, False-True)
def plot_bias_effects(df, error_type):
    fig, axes = plt.subplots(figsize=(10, 8))
    sns.boxplot(x='use_linear_bias', y=error_type, hue='use_constant_bias', data=df, ax=axes, order=[True, False])
    axes.set_title(f'Effect of Biases on {error_type}')
    axes.set_xlabel('use_linear_bias')
    axes.set_ylabel(error_type)
    plt.tight_layout()
    plt.show()

# Plot for position_error and full_anchor_error separately using filtered data (without outliers)
plot_bias_effects(filtered_data, 'position_error_linear')
plot_bias_effects(filtered_data, 'full_anchor_error_linear')

# Visualize the effects of KKR, IRLS, and LM across outlier probabilities
def plot_non_linear_methods_by_outlier_probability(df, error_type):
    outlier_probabilities = df['outlier_probability'].unique()
    methods = df['non_linear_method_rough'].unique()
    num_probabilities = len(outlier_probabilities)
    
    fig, axes = plt.subplots(num_probabilities, 1, figsize=(10, 8*num_probabilities), sharex=True)
    for i, outlier_prob in enumerate(outlier_probabilities):
        subset = df[df['outlier_probability'] == outlier_prob]
        sns.boxplot(x='non_linear_method_rough', y=error_type, data=subset, ax=axes[i], order=methods)
        axes[i].set_title(f'Outlier Probability = {outlier_prob}, {error_type}')
        axes[i].set_xlabel('Non-linear Method')
        axes[i].set_ylabel(error_type)
    plt.tight_layout()
    plt.show()

# Plot for position_error and full_anchor_error separately using filtered data (without outliers)
plot_non_linear_methods_by_outlier_probability(filtered_data, 'position_error')
plot_non_linear_methods_by_outlier_probability(filtered_data, 'full_anchor_error')


# Calculate error between final_estimator and anchor_gt
data['error_final'] = data.apply(lambda row: np.linalg.norm(row['final_estimator_components'][:3] - row['gt_anchor_components'][:3]), axis=1)
data['error_original'] = data.apply(lambda row: np.linalg.norm(row['original_estimator_components'][:3] - row['gt_anchor_components'][:3]), axis=1)

# Create a histogram of the error
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Histogram for error_final
axes[0].hist(data['error_final'], bins=np.logspace(np.log10(data['error_final'].min()), np.log10(data['error_final'].max()), 10), alpha=0.5, label='Final Estimator')
axes[0].set_xscale('log')
axes[0].set_xlabel('Error')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram of Error (Final Estimator)')

# Histogram for error_original
axes[1].hist(data['error_original'], bins=np.logspace(np.log10(data['error_original'].min()), np.log10(data['error_original'].max()), 10), alpha=0.5, label='Original Estimator')
axes[1].set_xscale('log')
axes[1].set_xlabel('Error')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Histogram of Error (Original Estimator)')

# Histogram for error_final/error_original
axes[2].hist(data['error_final']/data['error_original'], bins=np.linspace(0, 2, 50), alpha=0.5, label='Final Estimator / Original Estimator')
axes[2].set_xlabel('Error Ratio')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Histogram of Error Ratio (Final Estimator / Original Estimator)')

plt.tight_layout()
plt.show()