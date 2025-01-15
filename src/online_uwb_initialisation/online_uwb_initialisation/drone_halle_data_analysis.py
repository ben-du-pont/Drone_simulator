import csv
import numpy as np
import matplotlib.pyplot as plt

from online_uwb_initialisation.uwb_online_initialisation import UwbOnlineInitialisation

import csv
from pathlib import Path


package_path = Path(__file__).parent.resolve()
csv_dir = package_path / 'csv_files' / 'csv_drone_halle'

anchor_1_gt = [-2.460, -1.744, 0.233]
anchor_2_gt = [3.049, -1.385, 0.896]
anchor_3_gt = [-2.794, 1.067, 2.006]
anchor_4_gt = [2.861, 2.463, 2.143]

def process_csv(file_path):
    dataset_1 = []  # To store the first dataset
    dataset_2 = []  # To store the second dataset
    linear_estimate = []  # To store the linear estimate
    non_linear_estimate = []  # To store the non-linear estimate
    non_linear_final_estimate = []  # To store the final non-linear estimate

    # Read the file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        
        next(reader)  # Skip the header

        temp_dataset = []  # Temporary dataset to accumulate data
        segment = 1  # Tracks which part of the file we are processing

        for row in reader:
            # Strip any extra whitespace in the row to ensure accurate checking
            row = [r.strip() for r in row]

            # Skip empty rows
            if not row or all([not elem for elem in row]):
                # Handle segment transitions when encountering an empty row
                if segment == 1:
                    dataset_1 = temp_dataset  # Store the first dataset
                elif segment == 2:
                    linear_estimate = [float(val) for val in temp_dataset[0]]  # Linear estimate
                elif segment == 3:
                    non_linear_estimate = [float(val) for val in temp_dataset[0]]  # Non-linear estimate
                elif segment == 4:
                    dataset_2 = temp_dataset  # Store the second dataset
                elif segment == 5:
                    non_linear_final_estimate = [float(val) for val in temp_dataset[0]]  # Final non-linear estimate
                
                # Reset the temp dataset and move to next segment
                temp_dataset = []
                segment += 1
            else:
                # Convert all values in the row to float before appending
                temp_dataset.append([float(val) for val in row])  # Add non-empty row to the temp dataset

        # Handle the final segment if no empty line after the final estimate
        if segment == 4:
            dataset_2 = temp_dataset
        elif segment == 5:
            non_linear_final_estimate = [float(val) for val in temp_dataset[0]]

    return dataset_1, linear_estimate, non_linear_estimate, dataset_2, non_linear_final_estimate

def get_ground_truth(dataset):
    # Get the ground truth for the first anchor
    anchor_id = int(dataset[0][0])
    if anchor_id == 1:
        return anchor_1_gt
    elif anchor_id == 2:
        return anchor_2_gt
    elif anchor_id == 3:
        return anchor_3_gt
    elif anchor_id == 4:
        return anchor_4_gt
    else:
        raise ValueError('Invalid anchor ID')
    

def calculate_error(estimate, anchor_id):
    # Get the ground truth for the anchor
    if anchor_id == 1:
        ground_truth = anchor_1_gt
    elif anchor_id == 2:
        ground_truth = anchor_2_gt
    elif anchor_id == 3:
        ground_truth = anchor_3_gt
    elif anchor_id == 4:
        ground_truth = anchor_4_gt
    else:
        raise ValueError('Invalid anchor ID')
    
    # Calculate the error
    error = np.sqrt((estimate[0] - ground_truth[0]) ** 2 +
                    (estimate[1] - ground_truth[1]) ** 2 +
                    (estimate[2] - ground_truth[2]) ** 2)
    return error

def recompute_estimate(linear_estimate, dataset_1, method = "EM_new"):
    uwb_online = UwbOnlineInitialisation()
    uwb_online.params["non_linear_optimisation_type"] = method
    initial_guess = [float(linear_estimate[0]), float(linear_estimate[1]), float(linear_estimate[2]), 0, 1]

    measurements = []
    for row in dataset_1:
        id, range, x, y, z = row
        measurements.append([float(x), float(y), float(z), float(range)])
    print(measurements[0])
    estimate, cov = uwb_online.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess)

    return estimate

def recompute_final_estimate(non_linear_estimate, dataset_1, dataset_2, method="EM_new"):

    # Recompute the estimate using the UwbOnlineInitialisation class
    uwb_online = UwbOnlineInitialisation()
    uwb_online.params["non_linear_optimisation_type"] = method
    initial_guess = [float(non_linear_estimate[0]), float(non_linear_estimate[1]), float(non_linear_estimate[2]), 0, 1]

    measurements = []
    for row in dataset_1:
        id, range, x, y, z = row
        measurements.append([float(x), float(y), float(z), float(range)])
    for row in dataset_2:
        id, range, x, y, z = row
        measurements.append([float(x), float(y), float(z), float(range)])
    print(measurements[0])
    estimate, cov = uwb_online.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess)
    
    return estimate

def plot_data(dataset_1, dataset_2, linear_estimate, non_linear_estimate, non_linear_final_estimate):
    
    ground_truth = get_ground_truth(dataset_1)

    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111)

    dataset_full = dataset_1 + dataset_2

    # Calculate ground truth distances
    gt_distances = np.linalg.norm(np.array(dataset_full)[:, 2:5] - ground_truth, axis=1)
    estimated_distances = np.linalg.norm(np.array(dataset_full)[:, 2:5] - non_linear_final_estimate[:3], axis=1)
    rangespre = np.array(dataset_1)[:, 1]
    rangespost = np.array(dataset_2)[:, 1]

    # Plot ground truth and estimated distances
    ax.plot(gt_distances, 'b', label='Ground Truth range')
    ax.plot(estimated_distances, 'g', label='Expected range from estimate')
    
    # Plot ranges pre-deviation
    ax.plot(rangespre, 'ro', label='Ranges pre-deviation')

    # Plot ranges post-deviation with adjusted x-coordinates
    ax.plot(np.arange(len(rangespre), len(rangespre) + len(rangespost)), rangespost, 'ko', label='Ranges post-deviation')

    # Add legend and labels
    ax.set_xlabel('Measurement Count')
    ax.set_ylabel('Distance')
    ax.set_title('Expected and measured ranges')
    ax.legend()

def plot_3d_data(dataset_1, dataset_2, linear_estimate, non_linear_estimate, non_linear_final_estimate):
    ground_truth = get_ground_truth(dataset_1)

    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    dataset_full = dataset_1 + dataset_2
    dataset_full = np.array(dataset_full)

    # plot the measurements


    ax.plot(dataset_full[:, 2], dataset_full[:, 3], dataset_full[:, 4], 'r')

    # plot the ground truth
    ax.scatter(ground_truth[0], ground_truth[1], ground_truth[2], c='g', marker='x')

    # plot the linear estimate
    ax.scatter(linear_estimate[0], linear_estimate[1], linear_estimate[2], c='b', marker='x')

    # plot the non-linear estimate
    ax.scatter(non_linear_estimate[0], non_linear_estimate[1], non_linear_estimate[2], c='y', marker='x')

    # plot the final non-linear estimate
    ax.scatter(non_linear_final_estimate[0], non_linear_final_estimate[1], non_linear_final_estimate[2], c='m', marker='x')

    # for row in dataset_full:
    #     id, range, x, y, z = row
    #     ax.scatter(x, y, z, c='r', marker='o')

    plt.legend(['Measurements', 'Ground Truth', 'Linear Estimate', 'Non-Linear Estimate', 'Final Non-Linear Estimate'])
    plt.show()

def compute_position(range_1, range_2, range_3, range_4, anchor_1_gt, anchor_2_gt, anchor_3_gt, anchor_4_gt):
    """
    Computes the 3D position of the object using trilateration given 4 range measurements 
    and the positions of 4 known anchors.
    
    Arguments:
    - range_1, range_2, range_3, range_4: Distances from the object to each anchor.
    - anchor_1_gt, anchor_2_gt, anchor_3_gt, anchor_4_gt: Ground truth (x, y, z) positions of the anchors.
    
    Returns:
    - estimated_position: The estimated (x, y, z) position of the object.
    """
    measurement_vector = np.array([[*anchor_1_gt, range_1], [*anchor_2_gt, range_2], [*anchor_3_gt, range_3], [*anchor_4_gt, range_4]])

    anchor_estimator, cov, res, x = UwbOnlineInitialisation().estimate_anchor_position_linear_least_squares(measurement_vector)
    estimated_position = anchor_estimator[:3]
    
    return estimated_position
    
def compute_drone_position(data_anchor_1, data_anchor_2, data_anchor_3, data_anchor_4, anchor_1_gt, anchor_2_gt, anchor_3_gt, anchor_4_gt, anchor_1_estimate, anchor_2_estimate, anchor_3_estimate, anchor_4_estimate):
    data_set = [data_anchor_1, data_anchor_2, data_anchor_3, data_anchor_4]
    ground_truth = [anchor_1_gt, anchor_2_gt, anchor_3_gt, anchor_4_gt]

    longest_data_set = max(data_set, key=len)
    shortest_data_set = min(data_set, key=len)

    drone_positions = []
    drone_positions_gt = []
    drone_positions_estimated = []
    for i in range(len(shortest_data_set)-1):
        range_1 = data_anchor_1[i][1]
        range_2 = data_anchor_2[i][1]
        range_3 = data_anchor_3[i][1]
        range_4 = data_anchor_4[i][1]
        ranges = [range_1, range_2, range_3, range_4]
        print("RANGES",ranges)
        position_1 = data_anchor_1[i][2:]
        position_2 = data_anchor_2[i][2:]
        position_3 = data_anchor_3[i][2:]
        position_4 = data_anchor_4[i][2:]
        positions = [position_1, position_2, position_3, position_4]
        print("POSITIONS",positions)

        mean_position = np.mean([position_1, position_2, position_3, position_4], axis=0)

        drone_positions_gt.append(mean_position)
        # Compute the position of the drone
        drone_position = compute_position(range_1, range_2, range_3, range_4, anchor_1_gt, anchor_2_gt, anchor_3_gt, anchor_4_gt)
        drone_positions.append(drone_position)

        drone_position_estimate = compute_position(range_1, range_2, range_3, range_4, anchor_1_estimate, anchor_2_estimate, anchor_3_estimate, anchor_4_estimate)
        drone_positions_estimated.append(drone_position_estimate)

    return drone_positions, drone_positions_gt, drone_positions_estimated


# run 6.1 is nice to show
# run 4.1 is the high error one
# run 8.1 is the one with the lowest error
csv_file = csv_dir / 'run_8_2_results.csv'

# Usage
file_path = csv_file
dataset_1, linear_estimate, non_linear_estimate, dataset_2, non_linear_final_estimate = process_csv(file_path)

# Example outputs:
print("Dataset 1:", dataset_1)
print("Linear Estimate:", linear_estimate)
print("Non-Linear Estimate:", non_linear_estimate)
print("Dataset 2:", dataset_2)
print("Non-Linear Final Estimate:", non_linear_final_estimate)

new_non_linear_guess = recompute_estimate(linear_estimate, dataset_1)
new_final_guess = recompute_final_estimate(new_non_linear_guess, dataset_1, dataset_2)

non_linear_guess_IRLS = recompute_estimate(linear_estimate, dataset_1, method="IRLS")
final_guess_IRLS = recompute_final_estimate(non_linear_guess_IRLS, dataset_1, dataset_2, method="IRLS")

print("Previous non-linear estimate: ", non_linear_estimate)
print("New non-linear estimate: ", new_non_linear_guess)
print("Previous non-linear guess error", calculate_error(non_linear_estimate, int(dataset_1[0][0])))
print("New non-linear guess IRLS error", calculate_error(non_linear_guess_IRLS, int(dataset_1[0][0])))
print("New non-linear guess error", calculate_error(new_non_linear_guess, int(dataset_1[0][0])))

print("")

print("Previous final estimate: ", non_linear_final_estimate)
print("New final estimate: ", new_final_guess)
print("Previous final guess error", calculate_error(non_linear_final_estimate, int(dataset_1[0][0])))
print("New final guess IRLS error", calculate_error(final_guess_IRLS, int(dataset_1[0][0])))
print("New final guess error", calculate_error(new_final_guess, int(dataset_1[0][0])))

print("Ground truth: ", get_ground_truth(dataset_1))

plot_data(dataset_1, dataset_2, linear_estimate, new_non_linear_guess, new_final_guess)
plot_3d_data(dataset_1, dataset_2, linear_estimate, new_non_linear_guess, new_final_guess)

def process_multiple_runs(csv_dir, run_range=(4, 8), anchor_range=(1, 4)):
    linear_errors = {1: [], 2: [], 3: [], 4: []}
    non_linear_errors = {1: [], 2: [], 3: [], 4: []}
    final_errors = {1: [], 2: [], 3: [], 4: []}
    final_errors_irls = {1: [], 2: [], 3: [], 4: []}
    for run in range(run_range[0], run_range[1] + 1):
        for anchor in range(anchor_range[0], anchor_range[1] + 1):
            csv_file = csv_dir / f'run_{run}_{anchor}_results.csv'

            # Check if the file exists before processing
            if csv_file.exists():
                print(f"Processing {csv_file}")
                # Process the CSV file
                dataset_1, linear_estimate, non_linear_estimate, dataset_2, non_linear_final_estimate = process_csv(csv_file)

                # Recompute estimates
                new_non_linear_guess = recompute_estimate(linear_estimate, dataset_1)
                new_final_guess = recompute_final_estimate(new_non_linear_guess, dataset_1, dataset_2)
                final_guess_IRLS = recompute_final_estimate(new_non_linear_guess, dataset_1, dataset_2, method="IRLS")
                # Get the anchor ID and ground truth
                anchor_id = int(dataset_1[0][0])

                # Compute the errors for each estimate
                linear_error = calculate_error(linear_estimate, anchor_id)
                non_linear_error = calculate_error(new_non_linear_guess, anchor_id)
                final_error = calculate_error(new_final_guess, anchor_id)
                final_error_irls = calculate_error(final_guess_IRLS, anchor_id)
                # Append errors to corresponding error lists
                linear_errors[anchor_id].append(linear_error)
                non_linear_errors[anchor_id].append(non_linear_error)
                final_errors[anchor_id].append(final_error)
                final_errors_irls[anchor_id].append(final_error_irls)
            else:
                print(f"File {csv_file} not found, skipping.")

    return linear_errors, non_linear_errors, final_errors, final_errors_irls

def plot_boxplots(linear_errors, non_linear_errors, final_errors_irls, final_errors):
    # Combine the error lists into sets for boxplotting
    anchors = [1, 2, 3, 4]
    labels = ['Anchor 1', 'Anchor 2', 'Anchor 3', 'Anchor 4']

    # Plotting the boxplots side by side for linear, non-linear, final, and IRLS errors
    plt.figure(figsize=(20, 6))

    # Define the thresholds for shading
    thresholds = {
        'red': 0.5,
        'orange': 0.35,
        'green': 0.2
    }

    # Boxplot for Linear Errors
    plt.subplot(1, 4, 1)
    plt.boxplot([1.5*np.array(linear_errors[anchor]) for anchor in anchors], labels=labels)
    plt.title('Linear Estimate Errors')
    plt.ylabel('Error (meters)')
    plt.xlabel('Anchor ID')
    # Shading under the lines for Linear Errors
    plt.axhline(y=thresholds['red'], color='r', linestyle='--')
    plt.axhline(y=thresholds['orange'], color='orange', linestyle='--')
    plt.axhline(y=thresholds['green'], color='g', linestyle='--')
    plt.ylim(0, 2.1)

    # Fill between the thresholds for Linear Errors
    plt.fill_betweenx(y=[thresholds['green'], thresholds['orange']], x1=0.5, x2=4.5, color='green', alpha=0.1)
    plt.fill_betweenx(y=[thresholds['orange'], thresholds['red']], x1=0.5, x2=4.5, color='orange', alpha=0.1)
    plt.fill_betweenx(y=[thresholds['red'], 3.5], x1=0.5, x2=4.5, color='red', alpha=0.1)

    # Boxplot for Non-linear Errors
    plt.subplot(1, 4, 2)
    plt.boxplot([non_linear_errors[anchor] for anchor in anchors], labels=labels)
    plt.title('Non-Linear Estimate Errors (GMM)')
    plt.ylabel('Error (meters)')
    plt.xlabel('Anchor ID')
    # Shading under the lines for Non-linear Errors
    plt.axhline(y=thresholds['red'], color='r', linestyle='--')
    plt.axhline(y=thresholds['orange'], color='orange', linestyle='--')
    plt.axhline(y=thresholds['green'], color='g', linestyle='--')

    # Fill between the thresholds for Non-linear Errors
    plt.fill_betweenx(y=[thresholds['green'], thresholds['orange']], x1=0.5, x2=4.5, color='green', alpha=0.1)
    plt.fill_betweenx(y=[thresholds['orange'], thresholds['red']], x1=0.5, x2=4.5, color='orange', alpha=0.1)
    plt.fill_betweenx(y=[thresholds['red'], 3.5], x1=0.5, x2=4.5, color='red', alpha=0.1)
    plt.ylim(0, 2.1)
    # Boxplot for Final Non-linear Errors
    plt.subplot(1, 4, 4)
    plt.boxplot([final_errors[anchor] for anchor in anchors], labels=labels)
    plt.title('Final Non-Linear Estimate Errors')
    plt.ylabel('Error (meters)')
    plt.xlabel('Anchor ID')
    # Shading under the lines for Final Non-linear Errors
    plt.axhline(y=thresholds['red'], color='r', linestyle='--')
    plt.axhline(y=thresholds['orange'], color='orange', linestyle='--')
    plt.axhline(y=thresholds['green'], color='g', linestyle='--')

    # Fill between the thresholds for Final Non-linear Errors
    plt.fill_betweenx(y=[thresholds['green'], thresholds['orange']], x1=0.5, x2=4.5, color='green', alpha=0.1)
    plt.fill_betweenx(y=[thresholds['orange'], thresholds['red']], x1=0.5, x2=4.5, color='orange', alpha=0.1)
    plt.fill_betweenx(y=[thresholds['red'], 3.5], x1=0.5, x2=4.5, color='red', alpha=0.1)
    plt.ylim(0, 2.1)
    # Boxplot for Final IRLS Errors
    plt.subplot(1, 4, 3)
    plt.boxplot([final_errors_irls[anchor] for anchor in anchors], labels=labels)
    plt.title('Final IRLS Estimate Errors')
    plt.ylabel('Error (meters)')
    plt.xlabel('Anchor ID')
    # Shading under the lines for Final IRLS Errors
    plt.axhline(y=thresholds['red'], color='r', linestyle='--')
    plt.axhline(y=thresholds['orange'], color='orange', linestyle='--')
    plt.axhline(y=thresholds['green'], color='g', linestyle='--')
    plt.ylim(0, 2.1)
    # Fill between the thresholds for Final IRLS Errors
    plt.fill_betweenx(y=[thresholds['green'], thresholds['orange']], x1=0.5, x2=4.5, color='green', alpha=0.1)
    plt.fill_betweenx(y=[thresholds['orange'], thresholds['red']], x1=0.5, x2=4.5, color='orange', alpha=0.1)
    plt.fill_betweenx(y=[thresholds['red'], 3.5], x1=0.5, x2=4.5, color='red', alpha=0.1)
    plt.ylim(0, 2.1)

    plt.tight_layout()
    plt.show()



# Run the error collection and plotting for the given runs and anchors
linear_errors, non_linear_errors, final_errors, final_errors_irls = process_multiple_runs(csv_dir, run_range=(4, 8), anchor_range=(1, 4))

# Plot boxplots for linear, non-linear, and final errors
plot_boxplots(linear_errors, non_linear_errors, final_errors_irls, final_errors)


dataset_1_1, linear_estimate_1, non_linear_estimate_1, dataset_2_1, non_linear_final_estimate_1 = process_csv(csv_dir / 'run_8_1_results.csv')
dataset_1_2, linear_estimate_2, non_linear_estimate_2, dataset_2_2, non_linear_final_estimate_2 = process_csv(csv_dir / 'run_8_2_results.csv')
dataset_1_3, linear_estimate_3, non_linear_estimate_3, dataset_2_3, non_linear_final_estimate_3 = process_csv(csv_dir / 'run_8_3_results.csv')
dataset_1_4, linear_estimate_4, non_linear_estimate_4, dataset_2_4, non_linear_final_estimate_4 = process_csv(csv_dir / 'run_8_4_results.csv')

dataset_1 = dataset_1_1 + dataset_2_1
dataset_2 = dataset_1_2 + dataset_2_2
dataset_3 = dataset_1_3 + dataset_2_3
dataset_4 = dataset_1_4 + dataset_2_4

anchor_1_estimate = anchor_1_gt + np.random.normal(0, 0.1, 3)
anchor_2_estimate = anchor_2_gt + np.random.normal(0, 0.1, 3)
anchor_3_estimate = anchor_3_gt + np.random.normal(0, 0.1, 3)
anchor_4_estimate = anchor_4_gt + np.random.normal(0, 0.1, 3)

drone_positions, drone_positions_gt, drone_positions_estimated = compute_drone_position(dataset_1, dataset_2, dataset_3, dataset_4, anchor_1_gt, anchor_2_gt, anchor_3_gt, anchor_4_gt, anchor_1_estimate, anchor_2_estimate, anchor_3_estimate, anchor_4_estimate)

# Plot the drone positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.array(drone_positions)[:, 0], np.array(drone_positions)[:, 1], np.array(drone_positions)[:, 2], 'r')
ax.plot(np.array(drone_positions_gt)[:, 0], np.array(drone_positions_gt)[:, 1], np.array(drone_positions_gt)[:, 2], 'g')
ax.plot(np.array(drone_positions_estimated)[:, 0], np.array(drone_positions_estimated)[:, 1], np.array(drone_positions_estimated)[:, 2], 'b')
plt.legend(['Estimated Drone Positions', 'Ground Truth Drone Positions', 'Estimated Drone Positions'])
plt.show()
