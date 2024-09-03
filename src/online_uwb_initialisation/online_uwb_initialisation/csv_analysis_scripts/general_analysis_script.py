import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

package_path = Path(__file__).parent.parent.resolve()
csv_dir = package_path / 'csv_files'
csv_path = csv_dir / 'linear_least_squares.csv'

# Load the data
data = pd.read_csv(csv_path, header=None)

random_variables = ['number_of_measurements', 'min_angle_between_two_consecutive_measurements', 'min_distance_to_anchor', 'max_distance_to_anchor', 'mean_distance_to_anchor', 'std_distance_to_anchor', 'angular_span_elevation', 'angular_span_azimuth', 'constant_bias_gt', 'linear_bias_gt', 'noise_variance_gt', 'measured_noise_var', 'measured_noise_mean', 'outlier_probability']
#error_metrics = ['linear_error', 'linear_error_full', 'linear_linear_bias_error', 'linear_constant_bias_error', 'error_lm', 'error_lm_full', 'constant_bias_error_lm', 'linear_bias_error_lm', 'error_irls', 'error_irls_full', 'constant_bias_error_irls' ,'linear_bias_error_irls', 'error_krr', 'error_krr_full','constant_bias_error_krr', 'linear_bias_error_krr']

settings1 = ["linear_no_bias", "linear_constant_bias", "linear_linear_bias", "linear_both_biases"]
settings2 = ["reweighted_linear_no_bias", "reweighted_linear_constant_bias", "reweighted_linear_linear_bias", "reweighted_linear_both_biases"]
settings3 = ["filtered_linear_no_bias", "filtered_linear_constant_bias", "filtered_linear_linear_bias", "filtered_linear_both_biases"]
settings4 = ["filtered_reweighted_linear_no_bias", "filtered_reweighted_linear_constant_bias", "filtered_reweighted_linear_linear_bias", "filtered_reweighted_linear_both_biases"]

settings = settings1 + settings2 + settings3 + settings4

error_metrics = []
for setting in settings:
    error_metrics += [f"{setting}_error", f"{setting}_error_full", f"{setting}_constant_bias_error", f"{setting}_linear_bias_error", f"{setting}_non_linear_error"]

data.columns = random_variables + error_metrics

def aggregate_data(df, groupby_columns, error_columns):
    """Aggregate the data by the specified random variables and calculate statistics for the errors."""
    grouped = df.groupby(groupby_columns).agg({col: ['mean', 'std', 'min', 'max'] for col in error_columns})
    grouped.columns = ['_'.join(col) for col in grouped.columns]  # Flatten the multi-index columns
    return grouped.reset_index()

def plot_error_vs_variable(df, random_variable, error_columns, plot_type='boxplot'):
    """Plot the errors as a function of a specified random variable, using subplots."""
    n_errors = len(error_columns)
    fig, axes = plt.subplots(n_errors, 1, figsize=(10, 6 * n_errors), sharex=True)
    
    for i, error_col in enumerate(error_columns):
        ax = axes[i]
        
        if plot_type == 'boxplot':
            sns.boxplot(x=random_variable, y=error_col, data=df, ax=ax)
        elif plot_type == 'scatter':
            sns.scatterplot(x=random_variable, y=error_col, data=df, ax=ax)
        elif plot_type == 'line':
            sns.lineplot(x=random_variable, y=error_col, data=df, ax=ax)
        
        ax.set_title(f'{error_col} vs {random_variable}')
        ax.set_xlabel(random_variable)
        ax.set_ylabel(error_col)
    
    plt.tight_layout()
    plt.show()

def analyze_simulation_results(df, groupby_columns, error_columns):
    """Load, aggregate, and plot simulation results."""
    
    # Step 2: Aggregate data
    aggregated_df = aggregate_data(df, groupby_columns, error_columns)
    
    # Step 3: Plot results
    for random_variable in groupby_columns:
        plot_error_vs_variable(df, random_variable, error_columns, plot_type='scatter')

def plot_overall_error_boxplot(df, error_columns, remove_outliers=False):
    """Plot boxplots for all error columns together with an option to remove outliers."""
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(data=df[error_columns], showfliers=not remove_outliers)
    plt.title('Comparison of All Error Types')
    plt.ylabel('Error Value')
    
    plt.tight_layout()
    plt.show()

# Example usage
error_columns = []
for setting in settings:
    error_columns += [f"{setting}_error", f"{setting}_non_linear_error"]
random_variables = random_variables # ['min_distance_to_anchor', 'max_distance_to_anchor', 'std_distance_to_anchor']
#error_columns = error_metrics # ['linear_error', 'error_lm', 'error_irls', 'error_krr']

plot_overall_error_boxplot(data, error_columns, remove_outliers=True)
plot_overall_error_boxplot(data, error_columns, remove_outliers=False)

# Analyze the simulation results
analyze_simulation_results(data, random_variables, error_columns)
