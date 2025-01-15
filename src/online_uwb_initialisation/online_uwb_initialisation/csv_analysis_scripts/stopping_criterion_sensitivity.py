import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

package_path = Path(__file__).parent.parent.resolve()
csv_dir = package_path / 'csv_files'
csv_path = csv_dir / 'criterion_sensitivity.csv'

def visualize_stopping_criteria(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path, header=None)

    def add_minimum_column(df, columns_to_compare, new_column_name):
        """
        Add a new column to the DataFrame which is the minimum of specified columns,
        or inf if any of those columns contain inf.
        
        Parameters:
        df : DataFrame
            The DataFrame to modify.
        columns_to_compare : list
            List of columns to compare for minimum.
        new_column_name : str
            The name for the new column.
        """
        
        # Create a mask for any columns with inf values
        inf_mask = df[columns_to_compare].isin([np.inf]).any(axis=1)
        
        # Compute the minimum across the specified columns
        min_values = df[columns_to_compare].min(axis=1)
        
        # Set the min_values to inf where the inf_mask is True
        min_values[inf_mask] = np.inf
        
        # Assign the computed values to the new column in the DataFrame
        df[new_column_name] = min_values

    # Define column names based on your data structure
    # Assuming the first 7 columns are thresholds and the rest are errors
    threshold_columns = ['GDOP_thresh', 'FIM_thresh', 'CondNum_thresh', 'Residuals_thresh', 
                         'Covariance_thresh', 'Verification_thresh', 'Convergence_count_thresh', 'Convergence_thresh']
    
    error_columns = ['GDOP_error', 'FIM_error', 'CondNum_error', 'Residuals_error', 
                     'Covariance_error', 'Verification_error', 'Convergence_error']
    
    # Assign column names to the DataFrame
    column_names = threshold_columns + error_columns
    df.columns = column_names

    # Add a new column for the minimum threshold value
    error_columns_to_compare = ['Covariance_error', 'Verification_error', 'Convergence_error']
    new_column_name = 'Covariance_verification_convergence'
    add_minimum_column(df, error_columns_to_compare, 'Covariance_verification_convergence')

    # Clean the DataFrame: Drop rows with NaN or inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
    df.dropna(subset=threshold_columns + error_columns, inplace=True)  # Drop rows with NaN in relevant columns

    # Plotting
    plt.figure(figsize=(18, 12))
    
    # Loop through each criterion to plot its errors against the randomized thresholds
    for i, error_col in enumerate(error_columns):
        plt.subplot(3, 3, i + 1)

        # Use regplot to plot the trend line without scatter points
        sns.regplot(data=df, x=threshold_columns[i], y=error_col, 
                    scatter=False, color='b', label='Trend Line', ci=95)  # Use 95% CI

        # Set logarithmic scale for the y-axis if needed
        plt.yscale('log')

        plt.title(f'{error_col.replace("_", " ").capitalize()} vs {threshold_columns[i].replace("_", " ").capitalize()}')
        plt.xlabel(threshold_columns[i].replace("_", " ").capitalize())
        plt.ylabel(error_col.replace("_", " ").capitalize())
        plt.grid()
        plt.legend()

    plt.subplot(3, 3, i + 2)

    # Use regplot to plot the trend line without scatter points
    sns.regplot(data=df, x=threshold_columns[i+1], y=error_col, 
                scatter=False, color='b', label='Trend Line', ci=95)  # Use 95% CI

    # Set logarithmic scale for the y-axis if needed
    plt.yscale('log')

    plt.title(f'{error_col.replace("_", " ").capitalize()} vs {threshold_columns[i].replace("_", " ").capitalize()}')
    plt.xlabel(threshold_columns[i+1].replace("_", " ").capitalize())
    plt.ylabel(error_col.replace("_", " ").capitalize())
    plt.grid()
    plt.legend()


    conservative_threshold = 1
    less_conservative_threshold = 3
    third_threshold = 5
    num_bins = 20
    plt.figure(figsize=(18, 6))

    for i, (threshold_col, error_col) in enumerate(zip(threshold_columns, error_columns)):
        plt.subplot(2, 4, i + 1)

        # Define bins based on the thresholds for the specific criterion
        bin_edges = np.linspace(df[threshold_col].min(), df[threshold_col].max(), num_bins + 1)

        # Count how many errors are above the thresholds for each bin
        counts_above_conservative = []
        counts_above_less_conservative = []
        counts_above_third_threshold = []
        
        for j in range(len(bin_edges) - 1):
            # Total number of elements in this bin
            total_in_bin = np.sum((df[threshold_col] > bin_edges[j]) & (df[threshold_col] <= bin_edges[j + 1]))

            if total_in_bin == 0:
                # If there are no data points in this bin, skip further calculations
                counts_above_conservative.append(0)
                counts_above_less_conservative.append(0)
                counts_above_third_threshold.append(0)
                continue

            # Count how many errors are above the conservative threshold
            count_conservative = np.sum((df[error_col] > conservative_threshold) & 
                                        (df[threshold_col] > bin_edges[j]) & 
                                        (df[threshold_col] <= bin_edges[j + 1]))
            counts_above_conservative.append(count_conservative / total_in_bin * 100)

            # Count how many errors are above the less conservative threshold
            count_less_conservative = np.sum((df[error_col] > less_conservative_threshold) & 
                                            (df[threshold_col] > bin_edges[j]) & 
                                            (df[threshold_col] <= bin_edges[j + 1]))
            counts_above_less_conservative.append(count_less_conservative / total_in_bin * 100)

            # Count how many errors are above the third threshold
            count_third = np.sum((df[error_col] > third_threshold) & 
                                (df[threshold_col] > bin_edges[j]) & 
                                (df[threshold_col] <= bin_edges[j + 1]))
            counts_above_third_threshold.append(count_third / total_in_bin * 100)

        # Convert the counts into numpy arrays for stacking
        percentages_above_conservative = np.array(counts_above_conservative)
        percentages_above_less_conservative = np.array(counts_above_less_conservative)
        percentages_above_third_threshold = np.array(counts_above_third_threshold)

        # Plotting the histogram with percentages of errors above the third threshold (least conservative, red)
        plt.bar(bin_edges[:-1], percentages_above_third_threshold, width=np.diff(bin_edges), 
                color='red', alpha=0.7, edgecolor='black', label=f'Above Estimate error: {third_threshold}')

        # Plotting the histogram with percentages of errors above less conservative threshold (orange)
        plt.bar(bin_edges[:-1], percentages_above_less_conservative - percentages_above_third_threshold, 
                bottom=percentages_above_third_threshold, width=np.diff(bin_edges),
                color='orange', alpha=0.7, edgecolor='black', label=f'Above Estimate error: {less_conservative_threshold}')

        # Plotting the histogram with percentages of errors above conservative threshold (yellow)
        plt.bar(bin_edges[:-1], percentages_above_conservative - percentages_above_less_conservative, 
                bottom=percentages_above_less_conservative, width=np.diff(bin_edges),
                color='yellow', alpha=0.7, edgecolor='black', label=f'Above Estimate error: {conservative_threshold}')

        plt.title(f'{error_col.replace("_", " ").capitalize()} Above Thresholds (%)')
        plt.xlabel(threshold_col.replace("_", " ").capitalize())
        plt.ylabel('Percentage (%)')
        plt.xticks(bin_edges, rotation=45)
        plt.ylim(0, 100)  # Set y limit to 100%
        plt.legend()

    plt.subplot(2, 4, 8)

    # Total count of entries in the new minimum error column
    total_count_min_error = len(df[new_column_name])

    # Count how many errors are above the three thresholds for the new minimum column
    count_conservative_min = np.sum(df[new_column_name] > conservative_threshold)
    count_less_conservative_min = np.sum(df[new_column_name] > less_conservative_threshold)
    count_third_min = np.sum(df[new_column_name] > third_threshold)

    # Calculate percentages
    percentage_conservative_min = count_conservative_min / total_count_min_error * 100
    percentage_less_conservative_min = count_less_conservative_min / total_count_min_error * 100
    percentage_third_min = count_third_min / total_count_min_error * 100

    # Stacked bar chart for the minimum error column with the three thresholds

    # Plot the least conservative threshold (red, smallest threshold at the bottom)
    plt.bar(['Min Error'], percentage_third_min, 
            color='red', alpha=0.7, edgecolor='black', label=f'Error above: {third_threshold}')

    # Plot the middle threshold (orange) on top of the red bar
    plt.bar(['Min Error'], percentage_less_conservative_min - percentage_third_min, 
            bottom=percentage_third_min, color='orange', alpha=0.7, edgecolor='black', label=f'Error above: {less_conservative_threshold}')

    # Plot the most conservative threshold (yellow) on top of the orange bar
    plt.bar(['Min Error'], percentage_conservative_min - percentage_less_conservative_min, 
            bottom=percentage_less_conservative_min, color='yellow', alpha=0.7, edgecolor='black', label=f'Error above: {conservative_threshold}')

    # Set title and labels
    plt.title('Combined criterions: Covariance - Verification vector - Convergence')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 20)  # Set y limit to 100%
    plt.legend()


    plt.tight_layout()
    plt.show()




# Usage
visualize_stopping_criteria(csv_path)
