import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

from pathlib import Path

package_path = Path(__file__).parent.parent.resolve()
csv_dir = package_path / 'csv_files'
csv_path = csv_dir / 'linear_least_squares.csv'

# Load the data
data = pd.read_csv(csv_path, header=None)
print(data.head())
data.columns = ['number_of_measurements', 'min_angle_between_two_consecutive_measurements','min_distance_to_anchor', 'max_distance_to_anchor', 'mean_distance_to_anchor', 'std_distance_to_anchor', 'angular_span_elevation', 'angular_span_azimuth', 'constant_bias_gt', 'linear_bias_gt', 'measured_noise_mean', 'noise_variance', 'measured_noise_var', 'outlier_probability', 'error_t_t', 'error_t_t_full', 'constant_bias_error_true_true', 'linear_bias_error_true_true', 'error_t_f', 'error_t_f_full', 'linear_bias_error_true_false', 'error_f_t', 'error_f_t_full', 'constant_bias_error_false_true', 'error_f_f', 'error_f_f_full']

# print("Length before filtering:", len(data))
# data = data[(data['error_t_t'] > 20) | (data['error_t_f'] > 20) | (data['error_f_t'] > 20) | (data['error_f_f'] > 20)]
# data = data[(data['error_t_t'] > 20) & (data['error_t_f'] > 20) & (data['error_f_t'] > 20) & (data['error_f_f'] > 20)]
# print("Length after filtering:", len(data))

print(data.head())  
### Position Error vs. Bias Configurations

# Define configurations
configs = ['error_t_t', 'error_t_f', 'error_f_t', 'error_f_f']
config_labels = ['True-True', 'True-False', 'False-True', 'False-False']

# Create a DataFrame for plotting
plot_data = pd.melt(data[configs], var_name='Configuration', value_name='Position Error')
plot_data['Configuration'] = plot_data['Configuration'].map(dict(zip(configs, config_labels)))

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Configuration', y='Position Error', data=plot_data)
plt.title('Position Error vs. Bias Configuration (Linear Bias / Constant Bias)')
plt.show()

### Estimator Error vs. Bias Configurations

# Define configurations for estimator errors
estimator_configs = ['error_t_t_full', 'error_t_f_full', 'error_f_t_full', 'error_f_f_full']
estimator_labels = ['True-True', 'True-False', 'False-True', 'False-False']

# Create a DataFrame for plotting
estimator_data = pd.melt(data[estimator_configs], var_name='Configuration', value_name='Estimator Error')
estimator_data['Configuration'] = estimator_data['Configuration'].map(dict(zip(estimator_configs, estimator_labels)))

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Configuration', y='Estimator Error', data=estimator_data)
plt.title('Estimator Error vs. Bias Configuration (Linear Bias / Constant Bias)')
plt.show()


### Bias Error Analysis
# NB: What wbout false true ?
# NB: Does it make sense to compare with an estimated mean noise ?
# NB: Compare the bias error with the true bias

plt.figure(figsize=(10, 6))
sns.scatterplot(x='measured_noise_mean', y='constant_bias_error_true_true', data=data)
plt.title('Constant Bias Error vs. Measured Noise Mean')
plt.xlabel('Measured Noise Mean')
plt.ylabel('Constant Bias Error')
plt.show()

# Define bias configurations and corresponding error columns
bias_configurations = {
    'Constant and Linear Biases': 'error_t_t',
    'Only Constant Bias': 'error_f_t',
    'Only Linear Bias': 'error_t_f',
    'No Biases': 'error_f_f'
}

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot histograms for each bias configuration
for ax, (title, error_col) in zip(axes, bias_configurations.items()):
    # Plot histogram for Measured Noise Mean
    sns.histplot(data=data, x='measured_noise_mean', bins=30, ax=ax, color='blue', kde=False)
    ax.set_title(f'{title} vs. Measured Noise Mean')
    ax.set_xlabel('Measured Noise Mean')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Create subplots for Noise Variance
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot histograms for each bias configuration
for ax, (title, error_col) in zip(axes, bias_configurations.items()):
    # Plot histogram for Noise Variance
    sns.histplot(data=data, x='noise_variance', bins=30, ax=ax, color='red', kde=False)
    ax.set_title(f'{title} vs. Noise Variance')
    ax.set_xlabel('Noise Variance')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.scatterplot(x='measured_noise_mean', y='linear_bias_error_true_true', data=data)
plt.title('Linear Bias Error vs. Measured Noise Mean')
plt.xlabel('Measured Noise Mean')
plt.ylabel('Linear Bias Error')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='noise_variance', y='constant_bias_error_true_true', data=data)
plt.title('Constant Bias Error vs. Noise Variance')
plt.xlabel('Noise Variance')
plt.ylabel('Constant Bias Error')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='noise_variance', y='linear_bias_error_true_true', data=data)
plt.title('Linear Bias Error vs. Noise Variance')
plt.xlabel('Noise Variance')
plt.ylabel('Linear Bias Error')
plt.show()

### Outlier impact
# NB: Use all the errors and not just the tt errors
plt.figure(figsize=(10, 6))
sns.scatterplot(x='outlier_probability', y='error_t_t', data=data)
plt.title('Position Error vs. Outlier Probability')
plt.xlabel('Outlier Probability')
plt.ylabel('Position Error')
plt.show()



# Create subplots for Measured Noise Mean vs. Error
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (title, error_col) in zip(axes, bias_configurations.items()):
    sns.regplot(x='measured_noise_mean', y=error_col, data=data, ax=ax, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
    ax.set_title(f'{title} - Error vs. Measured Noise Mean')
    ax.set_xlabel('Measured Noise Mean')
    ax.set_ylabel('Error')

plt.tight_layout()
plt.show()

# Create subplots for Noise Variance vs. Error
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (title, error_col) in zip(axes, bias_configurations.items()):
    sns.regplot(x='noise_variance', y=error_col, data=data, ax=ax, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
    ax.set_title(f'{title} - Error vs. Noise Variance')
    ax.set_xlabel('Noise Variance')
    ax.set_ylabel('Error')

plt.tight_layout()
plt.show()

# Create subplots for Outlier Probability vs. Error
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (title, error_col) in zip(axes, bias_configurations.items()):
    sns.regplot(x='outlier_probability', y=error_col, data=data, ax=ax, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
    ax.set_title(f'{title} - Position Error vs. Outlier Probability')
    ax.set_xlabel('Outlier Probability')
    ax.set_ylabel('Error')

plt.tight_layout()
plt.show()






bias_errors_data = pd.melt(data[['outlier_probability', 'constant_bias_error_true_true', 'linear_bias_error_true_true']],
                           id_vars=['outlier_probability'],
                           var_name='Bias Error Type',
                           value_name='Bias Error')


# Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='outlier_probability', y='Bias Error', hue='Bias Error Type', data=bias_errors_data, palette='viridis')
plt.title('Bias Errors vs. Outlier Probability')
plt.xlabel('Outlier Probability')
plt.ylabel('Bias Error')
plt.legend(title='Bias Error Type')
plt.show()

bias_errors = {
    'Linear Bias (with constant bias)': 'linear_bias_error_true_true',
    'Constant Bias (with linear bias)': 'constant_bias_error_true_true',
    'Linear Bias Only': 'linear_bias_error_true_false',
    'Constant Bias Only': 'constant_bias_error_false_true'
}

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
axes = axes.flatten()

# Plot for each bias configuration
for ax, (title, error_col) in zip(axes, bias_errors.items()):
    sns.regplot(x='outlier_probability', y=error_col, data=data, ax=ax, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
    ax.set_title(f'{title} - Bias Error vs. Outlier Probability')
    ax.set_xlabel('Outlier Probability')
    ax.set_ylabel('Bias Error')

plt.tight_layout()
plt.show()


### Distance to anchors
plt.figure(figsize=(10, 6))
sns.scatterplot(x='number_of_measurements', y='error_t_t', data=data)
plt.title('Position Error vs. Number of Measurements')
plt.xlabel('Number of Measurements')
plt.ylabel('Position Error')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='mean_distance_to_anchor', y='error_t_t', data=data)
plt.title('Position Error vs. Mean Distance to Anchor')
plt.xlabel('Mean Distance to Anchor')
plt.ylabel('Position Error')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='min_distance_to_anchor', y='error_t_t', data=data)
plt.title('Position Error vs. Minimum Distance to Anchor')
plt.xlabel('Minimum Distance to Anchor')
plt.ylabel('Position Error')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='max_distance_to_anchor', y='error_t_t', data=data)
plt.title('Position Error vs. Maximum Distance to Anchor')
plt.xlabel('Maximum Distance to Anchor')
plt.ylabel('Position Error')
plt.show()

# Variables to plot
variables = [
    'number_of_measurements',
    'mean_distance_to_anchor',
    'min_distance_to_anchor',
    'max_distance_to_anchor',
    'angular_span_elevation',
    'angular_span_azimuth',
    'min_angle_between_two_consecutive_measurements',
]

def plot_with_regression_lines(variable, error_col, ax, title):
    # Plot data and OLS regression line with confidence interval using Seaborn
    sns.regplot(x=variable, y=error_col, data=data, ax=ax, scatter_kws={'color':'blue'}, ci=95)

    # Prepare data for regression
    X = data[[variable]].values
    y = data[error_col].values

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Ridge Regression
    ridge = Ridge(alpha=10000.0)  # Adjust alpha (regularization strength) here
    ridge.fit(X_scaled, y)
    y_pred_ridge = ridge.predict(X_scaled)

    # Fit Kernel Ridge Regression with RBF kernel
    kernel_ridge = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)  # You can adjust alpha and gamma here
    kernel_ridge.fit(X_scaled, y)
    y_pred_kernel_ridge = kernel_ridge.predict(X_scaled)

    # Plot Ridge Regression line
    ax.plot(data[variable], y_pred_ridge, color='red', linewidth=2, label='Ridge Regression Line')
    
    # Plot Kernel Ridge Regression line
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx]
    X_sorted_scaled = scaler.transform(X_sorted)
    y_pred_kernel_ridge_sorted = kernel_ridge.predict(X_sorted_scaled)
    ax.plot(X_sorted, y_pred_kernel_ridge_sorted, color='green', linestyle='--', linewidth=2, label='Kernel Ridge Regression Line')

    # Set plot titles and labels
    ax.set_title(f'{title} - {variable.replace("_", " ").title()} vs. Position Error')
    ax.set_xlabel(variable.replace("_", " ").title())
    ax.set_ylabel('Position Error')
    ax.legend(loc='best')


# Create subplots for each variable with Ridge Regression for each bias configuration
for variable in variables:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (title, error_col) in zip(axes, bias_configurations.items()):
        plot_with_regression_lines(variable, error_col, ax, title)

    plt.tight_layout()
    plt.show()


### Error analysis with angular span
plt.figure(figsize=(10, 6))
sns.scatterplot(x='angular_span_elevation', y='error_t_t', data=data)
plt.title('Position Error vs. Angular Span Elevation')
plt.xlabel('Angular Span Elevation')
plt.ylabel('Position Error')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='angular_span_azimuth', y='error_t_t', data=data)
plt.title('Position Error vs. Angular Span Azimuth')
plt.xlabel('Angular Span Azimuth')
plt.ylabel('Position Error')
plt.show()


### Comparing True and Estimated Biases

plt.figure(figsize=(10, 6))
sns.scatterplot(x='constant_bias_gt', y='constant_bias_error_true_true', data=data)
plt.title('Constant Bias Error vs. True Constant Bias')
plt.xlabel('True Constant Bias')
plt.ylabel('Constant Bias Error')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='linear_bias_gt', y='linear_bias_error_true_true', data=data)
plt.title('Linear Bias Error vs. True Linear Bias')
plt.xlabel('True Linear Bias')
plt.ylabel('Linear Bias Error')
plt.show()


bias_configurations = {
    'Linear Bias (with constant bias)': 'linear_bias_error_true_true',
    'Linear Bias Only': 'linear_bias_error_true_false', 
    'Constant Bias (with linear bias)': 'constant_bias_error_true_true',
    'Constant Bias Only': 'constant_bias_error_false_true',
}

# Variables to plot
variables = [
    'constant_bias_gt',
    'linear_bias_gt'
]

# Function to perform Ridge and Kernel Ridge Regression and plot
def plot_with_regression_lines(variable, error_col, ax, title):
    # Plot data and OLS regression line with confidence interval using Seaborn
    sns.regplot(x=variable, y=error_col, data=data, ax=ax, scatter_kws={'color':'blue'}, ci=95)

    # Prepare data for regression
    X = data[[variable]].values
    y = data[error_col].values

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Ridge Regression
    ridge = Ridge(alpha=10000.0)  # Adjust alpha (regularization strength) here
    ridge.fit(X_scaled, y)
    y_pred_ridge = ridge.predict(X_scaled)

    # Fit Kernel Ridge Regression with RBF kernel
    kernel_ridge = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)  # Adjust alpha and gamma here
    kernel_ridge.fit(X_scaled, y)
    y_pred_kernel_ridge = kernel_ridge.predict(X_scaled)

    # Plot Ridge Regression line
    ax.plot(data[variable], y_pred_ridge, color='red', linewidth=2, label='Ridge Regression Line')
    
    # Plot Kernel Ridge Regression line
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx]
    X_sorted_scaled = scaler.transform(X_sorted)
    y_pred_kernel_ridge_sorted = kernel_ridge.predict(X_sorted_scaled)
    ax.plot(X_sorted, y_pred_kernel_ridge_sorted, color='green', linestyle='--', linewidth=2, label='Kernel Ridge Regression Line')

    # Set plot titles and labels
    ax.set_title(f'{title} - {variable.replace("_", " ").title()} vs. Bias Error')
    ax.set_xlabel(variable.replace("_", " ").title())
    ax.set_ylabel('Bias Error')
    ax.legend(loc='best')

bias_configurations = {
    'Linear Bias (with constant bias)': 'linear_bias_error_true_true',
    'Linear Bias Only': 'linear_bias_error_true_false', 
    'Constant Bias (with linear bias)': 'constant_bias_error_true_true',
    'Constant Bias Only': 'constant_bias_error_false_true',
}


# Function to create and save figures
def create_figures_for_bias_type(bias_type, configurations, variable):
    fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot only two configurations per figure
    for ax, (title, error_col) in zip(axes, configurations):

        print(f"Plotting {title} with column {error_col}")
        plot_with_regression_lines(variable, error_col, ax, title)

    fig.suptitle(f'{bias_type} Biases Analysis')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Select two configurations for each bias type
constant_bias_configurations = [(title, error_col) for title, error_col in bias_configurations.items() if 'constant_bias' in error_col]
linear_bias_configurations = [(title, error_col) for title, error_col in bias_configurations.items() if 'linear_bias' in error_col]


print(constant_bias_configurations)
print(linear_bias_configurations)

# Create figures
create_figures_for_bias_type('Constant', constant_bias_configurations, 'constant_bias_gt')
create_figures_for_bias_type('Linear', linear_bias_configurations, 'linear_bias_gt')


### Error Decomposition

# Create a DataFrame for error decomposition
decomp_data = data[['error_t_t', 'constant_bias_error_true_true', 'linear_bias_error_true_true']]
decomp_data = pd.melt(decomp_data, var_name='Error Type', value_name='Value')

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='Error Type', y='Value', data=decomp_data)
plt.title('Decomposition of Position Error')
plt.show()

### Error correlation

correlation_columns = [
    'number_of_measurements',
    'mean_distance_to_anchor',
    'min_distance_to_anchor',
    'max_distance_to_anchor',
    'measured_noise_mean',
    'noise_variance',
    'min_angle_between_two_consecutive_measurements',
    'outlier_probability',
    'constant_bias_error_true_true',
    'linear_bias_error_true_true',
    'linear_bias_error_true_false',
    'constant_bias_error_false_true',
    'error_t_t',
    'error_t_f',
    'error_f_t',
    'error_f_f'
]


correlation_matrix = data[correlation_columns].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Errors')
plt.show()


### Noise analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='measured_noise_mean', y='noise_variance', data=data)
plt.title('Measured Noise Mean vs. Noise Variance')
plt.xlabel('Measured Noise Mean')
plt.ylabel('Noise Variance')
plt.show()

### Additional Measurement Geometry Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='min_angle_between_two_consecutive_measurements', y='error_t_t', data=data)
plt.title('Position Error vs. Minimum Angle Between Consecutive Measurements')
plt.xlabel('Minimum Angle Between Two Consecutive Measurements')
plt.ylabel('Position Error')
plt.show()



