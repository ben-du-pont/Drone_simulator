from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns


package_path = Path(__file__).parent.resolve()
csv_dir = package_path

csv_path = csv_dir / 'error_values.csv'
df = pd.read_csv(csv_path)


print(df.keys())

# Scatter plots for each hyperparameter
hyperparameters = ['distance_to_anchor_threshold', 'gdop_threshold', 'weight_angle', 'weight_distance', 'weight_dev', 'num_measurements']
for hyperparameter in hyperparameters:
    plt.figure()
    plt.xlabel(hyperparameter)
    plt.ylabel('Error')
    plt.title(f'Error vs {hyperparameter}')
    plt.grid(True)
    
    # Remove outliers
    q1 = df['final_estimate_error'].quantile(0.25)
    q3 = df['final_estimate_error'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_df = df[(df['final_estimate_error'] >= lower_bound) & (df['final_estimate_error'] <= upper_bound)]
    
    sns.scatterplot(data=filtered_df, x=hyperparameter, y='final_estimate_error', color='red')
    plt.xscale('log')  # Set x-axis to log scale
    plt.show()

# # Pairplot to visualize the relationships between hyperparameters
# sns.pairplot(df[hyperparameters + ['final_estimate_error']])
# plt.show()

# Heatmap to visualize the correlation between hyperparameters and error
plt.figure()
sns.heatmap(df[hyperparameters + ['final_estimate_error']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Box plots to compare the distribution of error for each hyperparameter
for hyperparameter in hyperparameters:
    plt.figure()
    sns.boxplot(data=filtered_df, x=hyperparameter, y='final_estimate_error')
    plt.title(f'Error Distribution for {hyperparameter}')
    plt.show()

# Violin plots to compare the distribution of error for each hyperparameter
for hyperparameter in hyperparameters:
    plt.figure()
    sns.violinplot(data=filtered_df, x=hyperparameter, y='final_estimate_error')
    plt.title(f'Error Distribution for {hyperparameter}')
    plt.show()


average_error = df['final_estimate_error'].mean()
print(f"Average Final Estimator Error: {average_error}")

average_error = df['final_estimate_error'].median()
print(f"Median Final Estimator Error: {average_error}")