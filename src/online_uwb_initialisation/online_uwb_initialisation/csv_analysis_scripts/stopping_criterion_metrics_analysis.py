import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import ast 

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

package_path = Path(__file__).parent.parent.resolve()
csv_dir = package_path / 'csv_files'
csv_path = csv_dir / 'metrics.csv'




# Check from when the ground truth error is below 0.1 and stays below 0.1
# Look at the value of the GDOP at that point and the slope ?

# Average this for all the runs and get the std_dev

# Get the most stable crtierion to use


def convert_str_to_list(s):
    # Replace 'nan', 'inf', and '-inf' with their corresponding numpy constants
    s = s.replace('nan', 'np.nan').replace('inf', 'np.inf').replace('-inf', '-np.inf')
    # Evaluate the string as a Python expression and return the result
    try:
        return eval(s)
    except Exception as e:
        # If evaluation fails, return the original string
        return s

def elementwise_inverse(vector):
    # Convert to numpy array to handle element-wise operations
    vector_np = np.array(vector)
    # Avoid division by zero by replacing zero values with a very small number
    vector_np = np.where(vector_np == 0, np.nan, vector_np)
    return 1 / vector_np

def find_convergence_index(error_vector, threshold):
    """
    Find the index where the error first falls below the threshold and stays below it.
    Returns NaN if the error never stays below the threshold.
    """
    threshold = float(threshold)  # Ensure threshold is a float
    for idx in range(len(error_vector)):
        if all(e < threshold for e in error_vector[idx:]):
            return idx
    return np.nan  # Return NaN if the threshold is never met

def extract_criterion_value(criterion_vector, index):
    """
    Extract the criterion value at the specified index. Returns NaN if index is NaN.
    """
    if np.isnan(index) or index >= len(criterion_vector):
        return np.nan
    return criterion_vector[int(index)]

def calculate_criteria_at_thresholds(df, thresholds=[10, 5, 1]):
    """
    Calculate the criterion values at which the error falls below specified thresholds.
    """
    for threshold in thresholds:
        df[f'convergence_index_{threshold}'] = df['error_vector'].apply(lambda x: find_convergence_index(x, threshold))
        df[f'gdop_at_{threshold}'] = df.apply(lambda row: extract_criterion_value(np.array(row['gdop_vector']), row[f'convergence_index_{threshold}']), axis=1)
        df[f'fim_at_{threshold}'] = df.apply(lambda row: extract_criterion_value(np.array(row['inverse_fim_vector']), row[f'convergence_index_{threshold}']), axis=1)
        df[f'condition_number_at_{threshold}'] = df.apply(lambda row: extract_criterion_value(np.array(row['condition_number_vector']), row[f'convergence_index_{threshold}']), axis=1)
        df[f'residuals_at_{threshold}'] = df.apply(lambda row: extract_criterion_value(np.array(row['residuals_vector']), row[f'convergence_index_{threshold}']), axis=1)
        df[f'covariances_at_{threshold}'] = df.apply(lambda row: extract_criterion_value(np.array(row['covariances_vector']), row[f'convergence_index_{threshold}']), axis=1)

        # Calculate number of measurements to reach the threshold
        df[f'num_measurements_at_{threshold}'] = df[f'convergence_index_{threshold}'] + 1  # Index is zero-based, so add 1
    

    return df

# Read the data from the CSV file
data = pd.read_csv(csv_path, header=None, converters={i: convert_str_to_list for i in range(6)})
data.columns = ['gdop_vector', 'fim_vector', 'condition_number_vector', 'residuals_vector', 'covariances_vector','error_vector']

# Add a column equal toumn inverse of the fim_vector column
data['inverse_fim_vector'] = data['fim_vector'].apply(elementwise_inverse)

def find_nan_locations(df):
    """
    Finds the locations of NaN values in a DataFrame where each cell contains a numpy array.
    
    Parameters:
    df (pd.DataFrame): DataFrame where each cell contains a numpy array.
    
    Returns:
    List[Tuple[int, str, int]]: List of tuples containing (row_index, column_name, array_index) where NaN is found.
    """
    nan_locations = []
    
    def has_nan(array):
        """Check if the numpy array contains NaN."""
        return np.isnan(array).any()

    # Iterate over DataFrame rows and columns to locate NaNs
    for row_index, row in df.iterrows():
        for col_name, array in row.items():
            if has_nan(array):
                nan_indices = np.where(np.isnan(array))[0]
                for nan_index in nan_indices:
                    nan_locations.append((row_index, col_name, nan_index))
    
    return nan_locations

def remove_invalid_indices(df):
    """
    Removes indices with NaN or Inf values from all numpy arrays in a DataFrame
    for rows where any array contains NaNs or Infs.
    
    Parameters:
    df (pd.DataFrame): DataFrame where each cell contains a numpy array.
    
    Returns:
    pd.DataFrame: New DataFrame with indices removed from arrays in rows with NaNs or Infs.
    """
    
    def find_invalid_indices(array):
        """Find indices of NaN or Inf values in the numpy array."""
        return np.where(np.isnan(array) | np.isinf(array))[0]
    
    def clean_row(row):
        """Remove invalid indices from all arrays in a row."""
        # Collect all invalid indices across columns in the current row
        all_invalid_indices = set()
        for array in row:
            all_invalid_indices.update(find_invalid_indices(array))
        
        # Remove invalid indices from all arrays
        return [np.delete(array, list(all_invalid_indices)) for array in row]

    # Apply cleaning function to each row and reassemble DataFrame
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.apply(lambda row: clean_row(row), axis=1)
    
    # Rebuild DataFrame with cleaned rows
    cleaned_df = pd.DataFrame(cleaned_df.tolist(), columns=df.columns)
    
    return cleaned_df

nan_locations = find_nan_locations(data)

# print("NaN Locations:")
# print(nan_locations)
data = remove_invalid_indices(data)

nan_locations = find_nan_locations(data)

# print("NaN Locations:")
# print(nan_locations)


def find_nan_locations(df):
    """
    Finds the locations of NaN values in a DataFrame where each cell contains a numpy array.
    
    Parameters:
    df (pd.DataFrame): DataFrame where each cell contains a numpy array.
    
    Returns:
    List[Tuple[int, str, int]]: List of tuples containing (row_index, column_name, array_index) where NaN is found.
    """
    nan_locations = []
    
    def has_nan(array):
        """Check if the numpy array contains NaN."""
        return np.isnan(array).any()

    # Iterate over DataFrame rows and columns to locate NaNs
    for row_index, row in df.iterrows():
        for col_name, array in row.items():
            if has_nan(array):
                nan_indices = np.where(np.isnan(array))[0]
                for nan_index in nan_indices:
                    nan_locations.append((row_index, col_name, nan_index))
    
    return nan_locations

def plot_criteria_variance(df, thresholds=[10, 5, 1]):
    plt.figure(figsize=(18, 12))
    criteria = ['gdop', 'fim', 'condition_number', 'residuals', 'covariances']
    
    for i, criterion in enumerate(criteria):
        plt.subplot(2, 3, i + 1)
        data = []
        for threshold in thresholds:
            col_name = f'{criterion}_at_{threshold}'
            values = df[col_name].dropna()  # Drop NaN values
            data.extend([(threshold, val) for val in values])
        
        df_plot = pd.DataFrame(data, columns=['Threshold', f'{criterion}_value'])
        
        sns.boxplot(x='Threshold', y=f'{criterion}_value', data=df_plot)
        plt.title(f'{criterion.capitalize()} at Different Error Thresholds')

    # Plot the number of measurements needed
    plt.subplot(2, 3, len(criteria) + 1)
    num_measurements_data = []
    for threshold in thresholds:
        col_name = f'num_measurements_at_{threshold}'
        values = df[col_name].dropna()  # Drop NaN values
        num_measurements_data.extend([(threshold, val) for val in values])
    
    df_num_measurements = pd.DataFrame(num_measurements_data, columns=['Threshold', 'num_measurements'])
    
    sns.boxplot(x='Threshold', y='num_measurements', data=df_num_measurements)
    plt.title('Number of Measurements to Reach Error Thresholds')
    
    plt.tight_layout()
    plt.show()

data = calculate_criteria_at_thresholds(data, thresholds=[20, 10, 5, 2, 1])
plot_criteria_variance(data,thresholds=[20, 10, 5, 2, 1])

csv_index = 11 # Index of the run to plot for better visualisation

print(data.head())

for csv_index in range(0, len(data)):
    # Assuming you have data for metrics and ground truth error
    measurement_indices = range(len(data.iloc[csv_index]['gdop_vector']))

    # Create subplot layout
    plt.figure(figsize=(14, 10))

    # Subplot 1: FIM vs. Measurement Index
    plt.subplot(3, 2, 1)
    plt.plot(measurement_indices, data.iloc[csv_index]['inverse_fim_vector'], color='b', label='FIM')
    plt.yscale('log')
    plt.ylabel('FIM', color='b')
    plt.tick_params(axis='y', colors='b')



    plt.twinx()
    plt.plot(measurement_indices, data.iloc[csv_index]['error_vector'], color='r', label='Ground Truth Error')
    plt.yscale('log')
    plt.ylabel('Ground Truth Error', color='r')
    plt.tick_params(axis='y', colors='r')

    plt.title('FIM vs. Ground Truth Error')

    # Subplot 2: GDOP vs. Measurement Index
    plt.subplot(3, 2, 2)
    plt.plot(measurement_indices, data.iloc[csv_index]['gdop_vector'], color='b', label='GDOP')
    plt.yscale('log')
    plt.ylabel('GDOP', color='b')
    plt.tick_params(axis='y', colors='b')

    plt.twinx()
    plt.plot(measurement_indices, data.iloc[csv_index]['error_vector'], color='r', label='Ground Truth Error')
    plt.yscale('log')
    plt.ylabel('Ground Truth Error', color='r')
    plt.tick_params(axis='y', colors='r')

    plt.title('GDOP vs. Ground Truth Error')

    # Subplot 3: RMS Residuals vs. Measurement Index
    plt.subplot(3, 2, 3)
    plt.plot(measurement_indices, data.iloc[csv_index]['residuals_vector'], color='b', label='RMS Residuals')
    plt.yscale('log')
    plt.ylabel('RMS Residuals', color='b')
    plt.tick_params(axis='y', colors='b')

    plt.twinx()
    plt.plot(measurement_indices, data.iloc[csv_index]['error_vector'], color='r', label='Ground Truth Error')
    plt.yscale('log')
    plt.ylabel('Ground Truth Error', color='r')
    plt.tick_params(axis='y', colors='r')

    plt.title('RMS Residuals vs. Ground Truth Error')

    # Subplot 4: Condition Number vs. Measurement Index
    plt.subplot(3, 2, 4)
    plt.plot(measurement_indices, data.iloc[csv_index]['condition_number_vector'], color='b', label='Condition Number')
    plt.yscale('log')
    plt.ylabel('Condition Number', color='b')
    plt.tick_params(axis='y', colors='b')

    plt.twinx()
    plt.plot(measurement_indices, data.iloc[csv_index]['error_vector'], color='r', label='Ground Truth Error')
    plt.yscale('log')
    plt.ylabel('Ground Truth Error', color='r')
    plt.tick_params(axis='y', colors='r')

    plt.title('Condition Number vs. Ground Truth Error')

    # Subplot 5: Covariance Eigenvalues vs. Measurement Index
    plt.subplot(3, 2, 5)
    plt.plot(measurement_indices, data.iloc[csv_index]['covariances_vector'], color='b', label='Covariance Eigenvalues')
    plt.yscale('log')
    plt.ylabel('Covariance Eigenvalues', color='b')
    plt.tick_params(axis='y', colors='b')

    plt.twinx()
    plt.plot(measurement_indices, data.iloc[csv_index]['error_vector'], color='r', label='Ground Truth Error')
    plt.yscale('log')
    plt.ylabel('Ground Truth Error', color='r')
    plt.tick_params(axis='y', colors='r')

    plt.title('Covariance Eigenvalues vs. Ground Truth Error')

    plt.tight_layout()
    plt.show()




    plt.figure(figsize=(14, 10))
    plt.plot(measurement_indices, data.iloc[csv_index]['inverse_fim_vector'], color='b', label='FIM')
    plt.plot(measurement_indices, data.iloc[csv_index]['gdop_vector'], color='r', label='GDOP')
    plt.plot(measurement_indices, data.iloc[csv_index]['condition_number_vector'], color='g', label='Condition Number')
    plt.plot(measurement_indices, data.iloc[csv_index]['residuals_vector'], color='y', label='RMS Residuals')
    plt.plot(measurement_indices, data.iloc[csv_index]['covariances_vector'], color='m', label='Covariance Eigenvalues')
    plt.plot(measurement_indices, data.iloc[csv_index]['error_vector'], color='k', label='Ground Truth Error')
    plt.yscale('log')
    plt.legend()
    plt.show()












def extract_final_values(df, columns):
    final_values = {}
    
    for col in columns:
        final_values[col] = df[col].apply(lambda x: x[-1] if len(x) > 0 else np.nan)
    
    final_values['error'] = df['error_vector'].apply(lambda x: x[-1] if len(x) > 0 else np.nan)
    
    return pd.DataFrame(final_values)


metrics_columns = ['gdop_vector', 'inverse_fim_vector', 'condition_number_vector', 'residuals_vector', 'covariances_vector']
final_values_df = extract_final_values(data, metrics_columns)

def plot_metric_vs_error(df, metrics_columns):
    plt.figure(figsize=(18, 12))
    
    for i, col in enumerate(metrics_columns):
        plt.subplot(2, 3, i + 1)
        sns.scatterplot(data=df, x=col, y='error')
        sns.regplot(data=df, x=col, y='error', scatter=False, color='red')
        plt.title(f'{col} vs. Final Error')
        plt.xlabel(col)
        plt.ylabel('Final Error')

    plt.tight_layout()
    plt.show()

plot_metric_vs_error(final_values_df, metrics_columns)



def compute_correlations(df, metrics_columns):
    correlations = {}
    
    for col in metrics_columns:
        pearson_corr = df[[col, 'error']].corr(method='pearson').iloc[0, 1]
        spearman_corr = df[[col, 'error']].corr(method='spearman').iloc[0, 1]
        correlations[col] = {
            'Pearson': pearson_corr,
            'Spearman': spearman_corr
        }
    
    return correlations

correlations = compute_correlations(final_values_df, metrics_columns)
print("Correlations between metrics and final error:")
for metric, corr in correlations.items():
    print(f"{metric}: Pearson={corr['Pearson']:.3f}, Spearman={corr['Spearman']:.3f}")


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def fit_and_plot_regression(df, metrics_columns):
    plt.figure(figsize=(18, 12))
    
    for i, col in enumerate(metrics_columns):
        X = df[[col]].values.reshape(-1, 1)
        y = df['error'].values

        # Linear regression
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        plt.subplot(2, 3, i + 1)
        plt.scatter(df[col], df['error'], color='blue', alpha=0.5, label='Data')
        plt.plot(df[col], predictions, color='red', linewidth=2, label='Linear Fit')
        plt.title(f'{col} vs. Final Error')
        plt.xlabel(col)
        plt.ylabel('Final Error')
        plt.legend()

    plt.tight_layout()
    plt.show()

fit_and_plot_regression(final_values_df, metrics_columns)




metrics_columns = ['gdop', 'fim', 'condition_number', 'residuals', 'covariances']




def plot_metric_heatmaps(df, metrics_columns, thresholds):
    plt.figure(figsize=(18, 12))
    
    for i, threshold in enumerate(thresholds):
        plt.subplot(len(thresholds), 1, i + 1)
        
        heatmap_data = {}
        for metric in metrics_columns:
            metric_values = df[f'{metric}_at_{threshold}'].dropna()
            error_values = df['error_vector'].dropna()
            heatmap_data[metric] = metric_values
        
        heatmap_df = pd.DataFrame(heatmap_data)
        sns.heatmap(heatmap_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Correlation Heatmap of Metrics at Error Threshold {threshold}')
    
    plt.tight_layout()
    plt.show()

plot_metric_heatmaps(data, metrics_columns, thresholds=[20, 10, 5, 2, 1])




def plot_pairwise_comparisons(df, metrics_columns, thresholds):
    plt.figure(figsize=(18, 12))
    
    for i, metric1 in enumerate(metrics_columns):
        for j, metric2 in enumerate(metrics_columns):
            if i < j:
                plt.subplot(len(metrics_columns), len(metrics_columns), i * len(metrics_columns) + j)
                
                df_plot = df[[f'{metric1}_at_{thresholds[-1]}', f'{metric2}_at_{thresholds[-1]}']].dropna()
                plt.scatter(df_plot[f'{metric1}_at_{thresholds[-1]}'], df_plot[f'{metric2}_at_{thresholds[-1]}'], alpha=0.5)
                
                plt.xlabel(metric1)
                plt.ylabel(metric2)
                plt.title(f'{metric1} vs. {metric2}')
    
    plt.tight_layout()
    plt.show()

plot_pairwise_comparisons(data, metrics_columns, thresholds=[20, 10, 5, 2, 1])








def evaluate_stopping_criteria(df, metrics, thresholds):
    results = {}
    for metric in metrics:
        results[metric] = {}
        for threshold in thresholds:
            filtered_df = df[df[f'{metric}_at_{threshold}'].notna()]
            success_rate = (filtered_df['error_vector'].apply(lambda x: np.any(np.array(x) < threshold)).mean())
            results[metric][threshold] = success_rate
    return pd.DataFrame(results)

thresholds=[20, 10, 5, 2, 1]
stopping_criteria_results = evaluate_stopping_criteria(data, metrics_columns, thresholds)

def plot_comparative_analysis(results):
    plt.figure(figsize=(12, 8))
    results.plot(kind='bar')
    plt.title('Stopping Criterion Effectiveness')
    plt.xlabel('Metric')
    plt.ylabel('Success Rate')
    plt.legend(title='Threshold')
    plt.tight_layout()
    plt.show()

plot_comparative_analysis(stopping_criteria_results)






