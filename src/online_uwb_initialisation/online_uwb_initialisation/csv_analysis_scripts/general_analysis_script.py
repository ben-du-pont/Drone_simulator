import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.optimize import curve_fit
from pathlib import Path

package_path = Path(__file__).parent.parent.resolve()
csv_dir = package_path / 'csv_files'
csv_path = csv_dir / 'monte_carlo_drone_halle.csv'
# csv_path = csv_dir / 'linear_least_squares.csv'

# Load the data
data = pd.read_csv(csv_path, header=None)

random_variables = ['number_of_measurements', 'min_angle_between_two_consecutive_measurements', 'min_distance_to_anchor', 'max_distance_to_anchor', 'mean_distance_to_anchor', 'std_distance_to_anchor', 'angular_span_elevation', 'angular_span_azimuth', 'constant_bias_gt', 'linear_bias_gt', 'noise_variance_gt', 'measured_noise_var', 'measured_noise_mean', 'outlier_probability']
#error_metrics = ['linear_error', 'linear_error_full', 'linear_linear_bias_error', 'linear_constant_bias_error', 'error_lm', 'error_lm_full', 'constant_bias_error_lm', 'linear_bias_error_lm', 'error_irls', 'error_irls_full', 'constant_bias_error_irls' ,'linear_bias_error_irls', 'error_krr', 'error_krr_full','constant_bias_error_krr', 'linear_bias_error_krr']

settings1 = ["linear_no_bias", "linear_constant_bias", "linear_linear_bias", "linear_both_biases"]
settings2 = ["reweighted_linear_no_bias", "reweighted_linear_constant_bias", "reweighted_linear_linear_bias", "reweighted_linear_both_biases"]
settings3 = ["filtered_linear_no_bias", "filtered_linear_constant_bias", "filtered_linear_linear_bias", "filtered_linear_both_biases"]
settings4 = ["filtered_reweighted_linear_no_bias", "filtered_reweighted_linear_constant_bias", "filtered_reweighted_linear_linear_bias", "filtered_reweighted_linear_both_biases"]

settings = settings1 + settings2 + settings3 + settings4

settings = ["reweighted_linear_no_bias", "reweighted_linear_constant_bias", "reweighted_linear_linear_bias", "reweighted_linear_both_biases", "regularised_weighted_linear_no_bias","regularised_weighted_linear_constant_bias", "regularised_weighted_linear_linear_bias", "regularised_weighted_linear_both_biases"]

settings = ["regularised_weighted_linear_constant_bias"]



error_metrics = []
for setting in settings:
    error_metrics += [f"{setting}_error", f"{setting}_error_full", f"{setting}_constant_bias_error", f"{setting}_linear_bias_error", f"{setting}_non_linear_error"]

error_metrics = ['linear_error', 'error_lm', 'error_irls', 'error_mm', 'error_em']


# data.columns = random_variables + error_metrics
template_row_anchor = ['noise','linear_error', 'non_linear_error', 'final_error']
template_error_row = ['linear_error', 'non_linear_error', 'final_error']
# make this for 4 anchors
full_row = [f"{col}_{str(id+1)}"  for id in range(4) for col in template_row_anchor]
error_columns = [f"{col}_{str(id+1)}"  for id in range(4) for col in template_error_row]
data.columns = full_row

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

def plot_overall_error_boxplot_linear(df, error_columns, remove_outliers=False):
    """Plot boxplots for all error columns together with an option to remove outliers."""
    plt.figure(figsize=(12, 8))
    
    
    plt.title('Comparison of the least square formulations')
    plt.ylabel('Error in anchor estimation')
    
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  +['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  + ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  +['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    legend_labels = ['No bias', 'Constant bias only', 'Linear bias only', 'Constant and linear bias']
    custom_labels = ['LS', 'WLS', 'LS - filtered', 'WLS - filtered'] +  ['LS', 'WLS', 'LS - filtered', 'WLS - filtered'] +  ['LS', 'WLS', 'LS - filtered', 'WLS - filtered'] +  ['LS', 'WLS', 'LS - filtered', 'WLS - filtered']
    sns.boxplot(data=df[error_columns], showfliers=not remove_outliers , palette=custom_colors)
    plt.xticks(ticks=range(len(error_columns)), labels=custom_labels, rotation=45)
    # Add the legend to the plot
    legend_colors = custom_colors[0:4]
    handles = [mpatches.Patch(color=legend_colors[i], label=legend_labels[i]) for i in range(4)]

    group_labels = ['Linear Least Squares', 'Weighted Linear Least Squares', 'Linear Least Squares - Filtered', 'Weighted Linear Least Squares - Filtered']
    group_positions = np.arange(1.5, len(custom_colors), 4)  # Position labels between every group of 4 plots
    plt.xticks(ticks=group_positions, labels=group_labels, rotation=0)

    plt.legend(handles=handles, title='Formulation')#, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_overall_error_boxplot_regularise(df, error_columns, remove_outliers=False):
    """Plot boxplots for all error columns together with an option to remove outliers."""
    plt.figure(figsize=(12, 8))
    
    
    plt.title('Comparison of the least square formulations')
    plt.ylabel('Error in anchor estimation')
    
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] + ['#ff7f0e', '#2ca02c', '#d62728']
    legend_labels = ['No bias', 'Constant bias only', 'Linear bias only', 'Constant and linear bias', 'Regularised constant bias only', 'Regularised linear bias only', 'Regularised constant and linear bias']
    custom_labels = ['No bias', 'Constant bias only', 'Linear bias only', 'Constant and linear bias', 'Regularised constant bias only', 'Regularised linear bias only', 'Regularised constant and linear bias']
    sns.boxplot(data=df[error_columns], showfliers=not remove_outliers , palette=custom_colors)
    plt.xticks(ticks=range(len(error_columns)), labels=custom_labels, rotation=45)
    # Add the legend to the plot
    legend_colors = custom_colors[0:4]
    handles = [mpatches.Patch(color=legend_colors[i], label=legend_labels[i]) for i in range(4)]

    group_labels = ['No regularisation', 'Regularised']
    group_positions = np.arange(1.5, 7, 4)  # Position labels between every group of 4 plots
    group_positions = [1.5, 5]
    plt.xticks(ticks=group_positions, labels=group_labels, rotation=0)

    plt.legend(handles=handles, title='Formulation')#, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_overall_error_boxplot_non_linear(df, error_columns, remove_outliers=False):
    """Plot boxplots for all error columns together with an option to remove outliers."""
    plt.figure(figsize=(12, 8))
    
    
    plt.title('Comparison of the least square formulations')
    plt.ylabel('Error in anchor estimation')
    
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  +['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  + ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  +['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] + ['#2ca02c', '#d62728']
    legend_labels = ['Error Linear', 'Error LM', 'Error IRLS', 'Error MM', 'Error GMM']

    custom_labels = ['Error Linear', 'Error LM', 'Error IRLS', 'Error MM', 'Error GMM']

    sns.boxplot(data=df[error_columns], showfliers=not remove_outliers)
    plt.xticks(ticks=range(len(error_columns)), labels=custom_labels, rotation=45)

    # Add the legend to the plot
    legend_colors = custom_colors[0:4]
    handles = [mpatches.Patch(color=legend_colors[i], label=legend_labels[i]) for i in range(4)]

    # group_labels = ['Linear Least Squares', 'Weighted Linear Least Squares', 'Linear Least Squares - Filtered', 'Weighted Linear Least Squares - Filtered']
    # group_positions = np.arange(1.5, len(custom_colors), 4)  # Position labels between every group of 4 plots
    # plt.xticks(ticks=group_positions, labels=group_labels, rotation=0)

    plt.legend(handles=handles, title='Formulation')#, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_with_regplot(df, error_column, groupby_column):
    plt.figure(figsize=(8, 6))

    # Create the regplot, specifying x and y
    sns.regplot(x=groupby_column, y=error_column, data=df, order=2, ci=95, scatter=False, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})

    # Customize plot labels
    plt.xlabel(groupby_column)
    plt.ylabel('Error')
    plt.title(f'Error vs {groupby_column} with Regression Line')

    # Show the plot
    plt.show()

def plot_multiple_regplots(df, error_columns, groupby_columns, shared_xlabel="Noise standard deviation"):
    # Check that the input lists have the same length
    assert len(error_columns) == len(groupby_columns), "The number of error columns must match the number of groupby columns."
    
    # Create a figure with subplots (1 column, multiple rows)
    fig, axs = plt.subplots(len(error_columns), 1, figsize=(8, 6 * len(error_columns)), sharex=True)
    
    # Loop through each error and groupby column
    for i, (error_column, groupby_column) in enumerate(zip(error_columns, groupby_columns)):
        ax = axs[i] if len(error_columns) > 1 else axs  # If only one subplot, axs is not a list

        # Apply square root transformation to the groupby column
        df['sqrt_' + groupby_column] = np.sqrt(df[groupby_column])

        # Create the regplot using the square root transformed x column
        sns.regplot(x='sqrt_' + groupby_column, y=error_column, data=df, order=2, ci=95, scatter=True, 
                    scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, ax=ax)
        
        # Customize the subplot labels
        ax.set_ylabel(f'Error in Estimation')
        ax.set_title(f'Error vs Noise standard deviation - anchor {i+1}', pad=10)  # Add padding to avoid overlap
        ax.set_xlabel('')
        ax.set_ylim(-0.5, 2)
    # Set the shared x-label for all subplots
    axs[-1].set_xlabel(f'Noise standard deviation')  # Set the x-label only on the last subplot (shared by all)
    
    # Adjust the layout to provide extra space between subplots
    plt.subplots_adjust(hspace=0.4)  # Increase hspace for more vertical space between plots
    
    # Show the figure
    plt.show()

def plot_overall_error_traj(df, error_columns, remove_outliers=False):
    """Plot boxplots for all error columns together with an option to remove outliers."""
    plt.figure(figsize=(12, 8))
    
    
    plt.title('Comparison of the error progression')
    plt.ylabel('Error in anchor estimation')
    
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  +['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  + ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  +['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] + ['#2ca02c', '#d62728']
    legend_labels = ['Error Linear', 'Error LM', 'Error IRLS', 'Error MM', 'Error GMM']

    custom_labels = ['Error Linear', 'Error Non-linear', 'Error FIM', 'Error GDOP']

    sns.boxplot(data=df[error_columns], showfliers=not remove_outliers, palette='Set2')
    
    plt.xticks(ticks=range(len(error_columns)), labels=custom_labels, rotation=45)

    # Add the legend to the plot
    legend_colors = custom_colors[0:4]
    handles = [mpatches.Patch(color=legend_colors[i], label=legend_labels[i]) for i in range(4)]

    # group_labels = ['Linear Least Squares', 'Weighted Linear Least Squares', 'Linear Least Squares - Filtered', 'Weighted Linear Least Squares - Filtered']
    # group_positions = np.arange(1.5, len(custom_colors), 4)  # Position labels between every group of 4 plots
    # plt.xticks(ticks=group_positions, labels=group_labels, rotation=0)

    # plt.legend(handles=handles, title='Formulation')#, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_overall_error_boxplot_monte_carlo(df, error_columns, remove_outliers=False):
    """
    Plot boxplots for error columns grouped by anchors, 
    with custom coloring for Linear, Non-Linear, and Final errors.
    """
    assert len(error_columns) % 3 == 0, "The number of error columns must be a multiple of 3 (Linear, Non-Linear, Final per anchor)."
    
    # Define custom colors for each type of error in a group (Linear, Non-Linear, Final)
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Linear (blue), Non-Linear (orange), Final (green)
    
    # Number of anchors (grouped by 3 errors per anchor)
    num_anchors = len(error_columns) // 3

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.title('Comparison of Error Progression by Anchor')
    plt.ylabel('Error in Anchor Estimation')

    # Create a color palette that repeats the custom colors for each anchor
    palette = sns.color_palette(custom_colors * num_anchors)

    # Plot the boxplot
    sns.boxplot(data=df[error_columns], showfliers=not remove_outliers, palette=palette)

    # Set custom x-axis labels for each group of 3 errors
    anchor_labels = []
    for i in range(1, num_anchors + 1):
        anchor_labels.extend([f'Anchor {i}', '', ''])
        anchor_mid_positions = [(i * 3 + 1) for i in range(num_anchors)]
    anchor_labels = [f'Anchor {i+1}' for i in range(num_anchors)]

    # Set the x-ticks at the middle of each group of 3 and apply the labels
    plt.xticks(ticks=anchor_mid_positions, labels=anchor_labels, rotation=0, ha='center')


    

    # Create the legend for the error types
    legend_labels = ['Linear Error', 'Non-Linear Error', 'Final Error']
    handles = [mpatches.Patch(color=custom_colors[i], label=legend_labels[i]) for i in range(3)]
    plt.legend(handles=handles, title='Error Type')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
# plot_with_regplot(data, error_column='regularised_weighted_linear_constant_bias_error', groupby_column='outlier_probability')
# plot_with_regplot(data, error_column='regularised_weighted_linear_constant_bias_error', groupby_column='noise_variance_gt')

plot_overall_error_boxplot_monte_carlo(data, error_columns, remove_outliers=True)
plot_multiple_regplots(data, error_columns= ["final_error_1", "final_error_2", "final_error_3", "final_error_4"], groupby_columns=['noise_1','noise_2','noise_3','noise_4'])

error_columns = []
for setting in settings:
    error_columns += [f"{setting}_error", f"{setting}_non_linear_error"]
random_variables = random_variables # ['min_distance_to_anchor', 'max_distance_to_anchor', 'std_distance_to_anchor']
# plot_variance_curve(data, error_columns, random_variables)


# # Example usage
# error_columns = []
# for setting in settings:
#     error_columns += [f"{setting}_error", f"{setting}_non_linear_error"]
random_variables = random_variables # ['min_distance_to_anchor', 'max_distance_to_anchor', 'std_distance_to_anchor']
error_columns = error_metrics # ['linear_error', 'error_lm', 'error_irls', 'error_krr']
plot_overall_error_boxplot_non_linear(data, error_columns, remove_outliers=False)
# settings = ["reweighted_linear_no_bias", "reweighted_linear_constant_bias", "reweighted_linear_linear_bias", "reweighted_linear_both_biases", "regularised_weighted_linear_constant_bias", "regularised_weighted_linear_linear_bias", "regularised_weighted_linear_both_biases"]


# error_columns = []
# for setting in settings:
#     error_columns += [f"{setting}_error"]
# # error_columns = ["linear_no_bias_error", "reweighted_linear_no_bias_error", "filtered_linear_no_bias_error", "filtered_reweighted_linear_no_bias_error"]

# plot_overall_error_boxplot_regularise(data, error_columns, remove_outliers=True)
# plot_overall_error_boxplot_regularise(data, error_columns, remove_outliers=False)

# # plot_overall_error_boxplot_linear(data, error_columns, remove_outliers=True)
# # plot_overall_error_boxplot_linear(data, error_columns, remove_outliers=False)


# # Analyze the simulation results
# analyze_simulation_results(data, random_variables, error_columns)
