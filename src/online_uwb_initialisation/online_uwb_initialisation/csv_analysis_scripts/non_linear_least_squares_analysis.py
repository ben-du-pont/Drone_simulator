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
csv_path = csv_dir / 'non_linear_least_squares.csv'

# Load the data
data = pd.read_csv(csv_path, header=None)

data.columns = ['number_of_measurements', 'min_angle_between_two_consecutive_measurements', 'min_distance_to_anchor', 'max_distance_to_anchor', 'mean_distance_to_anchor', 'std_distance_to_anchor', 'angular_span_elevation', 'angular_span_azimuth', 'constant_bias_gt', 'linear_bias_gt', 'measured_noise_mean', 'noise_variance', 'measured_noise_var', 'outlier_probability', 'linear_error', 'linear_error_full', 'linear_linear_bias_error', 'linear_constant_bias_error', 'error_lm', 'error_lm_full', 'constant_bias_error_lm', 'linear_bias_error_lm', 'error_irls', 'error_irls_full', 'constant_bias_error_irls' ,'linear_bias_error_irls', 'error_krr', 'error_krr_full','constant_bias_error_krr', 'linear_bias_error_krr']



# Define the columns for position errors from each nonlinear method
nonlinear_position_errors = {
    'LM': 'error_lm',
    'IRLS': 'error_irls',
    'KRR': 'error_krr'
}

# Create a DataFrame for plotting
plot_data_position_error = pd.melt(data[list(nonlinear_position_errors.values())], var_name='Method', value_name='Position Error')
plot_data_position_error['Method'] = plot_data_position_error['Method'].map(dict(zip(nonlinear_position_errors.values(), nonlinear_position_errors.keys())))

# Plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Method', y='Position Error', data=plot_data_position_error)
plt.title('Position Error Comparison for Nonlinear Optimization Methods')
plt.xticks(rotation=45)
plt.show()


# Define the columns for estimator errors from each nonlinear method
nonlinear_estimator_errors = {
    'LM - Full': 'error_lm_full',
    'IRLS - Full': 'error_irls_full',
    'KRR - Full': 'error_krr_full'
}

# Create a DataFrame for plotting
plot_data_estimator_error = pd.melt(data[list(nonlinear_estimator_errors.values())], var_name='Method', value_name='Estimator Error')
plot_data_estimator_error['Method'] = plot_data_estimator_error['Method'].map(dict(zip(nonlinear_estimator_errors.values(), nonlinear_estimator_errors.keys())))

# Plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Method', y='Estimator Error', data=plot_data_estimator_error)
plt.title('Estimator Error Comparison for Nonlinear Optimization Methods')
plt.xticks(rotation=45)
plt.show()



# Define the columns for constant and linear bias errors from each nonlinear method
nonlinear_bias_errors = {
    'LM - Constant Bias Error': 'constant_bias_error_lm',
    'LM - Linear Bias Error': 'linear_bias_error_lm',
    'IRLS - Constant Bias Error': 'constant_bias_error_irls',
    'IRLS - Linear Bias Error': 'linear_bias_error_irls',
    'KRR - Constant Bias Error': 'constant_bias_error_krr',
    'KRR - Linear Bias Error': 'linear_bias_error_krr'
}

# Create a DataFrame for plotting
plot_data_bias_error = pd.melt(data[list(nonlinear_bias_errors.values())], var_name='Method', value_name='Bias Error')
plot_data_bias_error['Method'] = plot_data_bias_error['Method'].map(dict(zip(nonlinear_bias_errors.values(), nonlinear_bias_errors.keys())))

# Plot
plt.figure(figsize=(14, 8))
sns.boxplot(x='Method', y='Bias Error', data=plot_data_bias_error)
plt.title('Bias Error Comparison for Nonlinear Optimization Methods')
plt.xticks(rotation=45)
plt.show()


# Define columns for correlation analysis
correlation_columns_nonlinear = [
    'number_of_measurements',
    'mean_distance_to_anchor',
    'min_distance_to_anchor',
    'max_distance_to_anchor',
    'measured_noise_mean',
    'noise_variance',
    'error_lm',
    'error_irls',
    'error_krr',
    'error_lm_full',
    'error_irls_full',
    'error_krr_full',
    'constant_bias_error_lm',
    'linear_bias_error_lm',
    'constant_bias_error_irls',
    'linear_bias_error_irls',
    'constant_bias_error_krr',
    'linear_bias_error_krr',
    
]

# Compute correlation matrix
correlation_matrix_nonlinear = data[correlation_columns_nonlinear].corr()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_nonlinear, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Nonlinear Errors')
plt.show()





# Define the outlier filtering function based on error values
def filter_outliers_by_error(df, error_col, lower_bound, upper_bound):
    """
    Filters outliers based on a specified range for error values.
    
    Parameters:
    - df: The DataFrame to filter.
    - error_col: The column containing the error values.
    - lower_bound: The lower bound for acceptable error values.
    - upper_bound: The upper bound for acceptable error values.
    
    Returns:
    - A DataFrame with outliers removed based on error values.
    """
    filtered_df = df[(df[error_col] >= lower_bound) & (df[error_col] <= upper_bound)]
    return filtered_df

# Define the plotting function with regression lines and error-based filtering
def plot_metric_impact(ax, df, metric, error_col, error_bounds):
    # Filter outliers based on error values
    df_filtered = filter_outliers_by_error(df, error_col, *error_bounds)
    
    # Scatter plot
    sns.scatterplot(x=metric, y=error_col, data=df_filtered, ax=ax, label=f'{error_col} Data')

    # Plot with Seaborn's regplot for OLS regression line
    sns.regplot(x=metric, y=error_col, data=df_filtered, ax=ax, scatter=False, line_kws={'color':'red'}, label=f'{error_col} OLS Regression Line')

    # Prepare data for Ridge and Kernel Ridge Regression
    X = df_filtered[[metric]].values
    y = df_filtered[error_col].values

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge Regression
    ridge = Ridge(alpha=10000.0)
    ridge.fit(X_scaled, y)
    y_pred_ridge = ridge.predict(X_scaled)

    # Kernel Ridge Regression with RBF kernel
    kernel_ridge = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
    kernel_ridge.fit(X_scaled, y)
    y_pred_kernel_ridge = kernel_ridge.predict(X_scaled)

    # Plot Ridge Regression line
    ax.plot(df_filtered[metric], y_pred_ridge, linewidth=2, label=f'{error_col} Ridge Regression Line')

    # Plot Kernel Ridge Regression line
    X_sorted = np.sort(X[:, 0])
    X_sorted_scaled = scaler.transform(X_sorted.reshape(-1, 1))
    y_pred_kernel_ridge_sorted = kernel_ridge.predict(X_sorted_scaled)
    ax.plot(X_sorted, y_pred_kernel_ridge_sorted, linestyle='--', linewidth=2, label=f'{error_col} Kernel Ridge Regression Line')

    # Set plot titles and labels
    ax.set_title(f'{metric.replace("_", " ").title()} vs {error_col.replace("_", " ").title()}')
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel('Error')
    ax.legend(loc='best')

# Define metrics to plot against
metrics = [
    'number_of_measurements',
    'min_angle_between_two_consecutive_measurements',
    'min_distance_to_anchor',
    'max_distance_to_anchor',
    'mean_distance_to_anchor',
    'std_distance_to_anchor',
    'angular_span_elevation',
    'angular_span_azimuth',
    'constant_bias_gt',
    'linear_bias_gt',
    'measured_noise_mean',
    'noise_variance',
    'measured_noise_var',
    'outlier_probability',
    'linear_error', 
    'linear_error_full', 
    'linear_linear_bias_error', 
    'linear_constant_bias_error'
]

# Error columns to compare against
error_cols = [
    'error_lm',
    'error_irls',
    'error_krr'
]

# Error bounds for filtering
error_bounds = {
    'error_lm': (data['error_lm'].quantile(0.05), data['error_lm'].quantile(0.95)),
    'error_irls': (data['error_irls'].quantile(0.05), data['error_irls'].quantile(0.95)),
    'error_krr': (data['error_krr'].quantile(0.05), data['error_krr'].quantile(0.95))
}


for metric in metrics:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex='col', sharey='row')
    for i, (error_col, ax) in enumerate(zip(error_cols, axes)):
        plot_metric_impact(ax, data, metric, error_col, error_bounds[error_col])
    plt.tight_layout()
    plt.suptitle(f'Impact of {metric.replace("_", " ").title()} on Position Errors', y=1.05)
    plt.show()
