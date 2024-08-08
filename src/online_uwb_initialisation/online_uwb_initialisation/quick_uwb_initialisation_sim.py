from drone_uwb_simulator.UWB_protocol import Anchor
from drone_uwb_simulator.drone_simulator import DroneSimulation
from drone_uwb_simulator.drone_dynamics import Waypoint, Trajectory

from online_uwb_initialisation.uwb_online_initialisation import UwbOnlineInitialisation
from online_uwb_initialisation.trajectory_optimisation import TrajectoryOptimization

import numpy as np
import csv
import json

import random
from scipy.spatial.transform import Rotation as R


from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import distance



import pandas as pd


package_path = Path(__file__).parent.resolve()
csv_dir = package_path / 'csv_files'

def array_to_string(array):
    return str(array).replace('\n', '').replace('[', '').replace(']', '')



class CalculateOnlineInitialisation:

    def __init__(self):
        """Initialise the CalculateOnlineInitialisation object."""

        # Create the DroneSimulation object, the UwbOnlineInitialisation object, and the TrajectoryOptimization object
        self.drone_sim = DroneSimulation()
        self.uwb_online_initialisation = UwbOnlineInitialisation()
        self.optimiser = TrajectoryOptimization()

        # Initialise the measurement vector and the ground truth measurement vector
        self.measurement_vector = []
        self.measurement_vector_gt = []

        # Initialise the vectors for the stopping criterion metrics
        self.gdop_vector = []
        self.fim_vector = []
        self.condition_number_vector = []
        self.covariances_vector = []
        self.sum_of_residuals_vector = []

        # Initialise the error vector
        self.error_vector = []


    def randomise_environment(self):
        """Function to randomise the environment by adjusting the anchor parameters and the measurement collection procedure."""

        # Randomise the anchor
        self.drone_sim.unknown_anchors[0].bias = random.uniform(0, 1)
        self.drone_sim.unknown_anchors[0].linear_bias = random.uniform(1, 1.5)
        self.drone_sim.unknown_anchors[0].noise_variance = random.uniform(0, 0.2)
        self.drone_sim.unknown_anchors[0].outlier_probability = random.uniform(0, 0.3)

        # Randomise the measurement collection delta
        self.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold'] = random.uniform(0.01, 0.05)

    def reset_metrics(self, anchor_id):
        """Reset the metrics vectors and the measurement vector.
        
        Parameters:
        anchor_id (int): The ID of the anchor for which the metrics are computed amd should be reset.
        """
        self.measurement_vector = []

        self.gdop_vector = []
        self.fim_vector = []
        self.condition_number_vector = []
        self.covariances_vector = []
        self.sum_of_residuals_vector = []

        self.uwb_online_initialisation.reset_all_measurements(anchor_id)

    def gather_measurements(self):
        """Function to gather measurements from the drone trajectory and the unknown anchors.
        The measurements are stored in the unknown_anchor_measurements dictionary in the UwbOnlineInitialisation object.
        They are also preprocessed, i.e only added if they are within the distance_to_anchor_ratio_threshold and close enough to the drone."""

        for p_x, p_y, p_z in zip(self.drone_sim.drone_trajectory.spline_x, self.drone_sim.drone_trajectory.spline_y, self.drone_sim.drone_trajectory.spline_z):
            _, unknown_anchor_distances = self.uwb_online_initialisation.get_distance_to_anchors(p_x, p_y, p_z)
            drone_position = [p_x, p_y, p_z]
            # Iterate over the unkown anchors in range
            for distance, anchor_id in unknown_anchor_distances: 
                self.uwb_online_initialisation.process_measurement(drone_position, distance, anchor_id)

    def gather_measurements_ground_truth(self):
        """Function to gather measurements from the drone trajectory and the unknown anchors, using the ground truth distance (no bias and no noise).
        The measurements are stored in the unknown_anchor_measurements dictionary in the UwbOnlineInitialisation object.
        They are also preprocessed, i.e only added if they are within the distance_to_anchor_ratio_threshold."""

        for p_x, p_y, p_z in zip(self.drone_sim.drone_trajectory.spline_x, self.drone_sim.drone_trajectory.spline_y, self.drone_sim.drone_trajectory.spline_z):
            _, unknown_anchor_distances = self.uwb_online_initialisation.get_distance_to_anchors_gt(p_x, p_y, p_z)
            drone_position = [p_x, p_y, p_z]
            # Iterate over the unkown anchors in range
            for distance, anchor_id in unknown_anchor_distances: 
                self.uwb_online_initialisation.process_measurement(drone_position, distance, anchor_id)






    def gather_metrics(self, anchor_id):
        """Run the pipeline on the measurements to iteratively estimate the anchor position and compute the stopping criterion variables.
        Store the stopping criterion metrics in the corresponding vectors.
        
        Parameters:
        anchor_id (int): The ID of the anchor for which the metrics are computed.
        """
        
        # Extract the drone positions and range measurements for the anchor from the unknown_anchor_measurements dictionary in the UwbOnlineInitialisation object
        drone_positions = self.uwb_online_initialisation.unknown_anchor_measurements[anchor_id]["positions_pre_rough_estimate"]
        range_measurements = self.uwb_online_initialisation.unknown_anchor_measurements[anchor_id]["distances_pre_rough_estimate"]

        # Run the pipleline
        for i in range(len(drone_positions)):

            # Append the measurement to the measurement vector
            self.measurement_vector.append([drone_positions[i][0], drone_positions[i][1], drone_positions[i][2], range_measurements[i]])

            # Check if there are enough measurements to run the linear least squares estimation and compute metrics
            if len(self.measurement_vector) > 5:

                # Run the linear least squares estimation to compute the estimate, the covariance matrix and the residuals
                estimator, covariance_matrix, residuals, _ = self.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(self.measurement_vector)

                # Compute the stopping criterion variables
                FIM = self.uwb_online_initialisation.compute_FIM(self.measurement_vector, estimator[:3])
                GDOP = self.uwb_online_initialisation.compute_GDOP(self.measurement_vector, estimator[:3])
                sum_of_residuals = np.mean(residuals)
                condition_number = self.uwb_online_initialisation.compute_condition_number(self.measurement_vector)
                covariances = np.diag(covariance_matrix)

                error = self.uwb_online_initialisation.calculate_position_error(0, estimator[:3])

                self.gdop_vector.append(GDOP)
                self.fim_vector.append(np.linalg.det(FIM))
                self.condition_number_vector.append(condition_number)
                self.covariances_vector.append(covariances)
                self.sum_of_residuals_vector.append(sum_of_residuals)
                self.error_vector.append(error)

            else: # If there are not enough measurements, append NaN values to the vectors
                self.gdop_vector.append(np.nan)
                self.fim_vector.append(np.nan)
                self.condition_number_vector.append(np.nan)
                self.covariances_vector.append([np.nan, np.nan, np.nan])
                self.sum_of_residuals_vector.append(np.nan)
                self.error_vector.append(np.nan)

    def gather_metrics_with_outlier_filtering(self, anchor_id):
        """Run the pipeline on the measurements to iteratively estimate the anchor position and compute the stopping criterion variables.
        Store the stopping criterion metrics in the corresponding vectors whilst removing outliers from the measurements."""

        # Extract the drone positions and range measurements for the anchor from the unknown_anchor_measurements dictionary in the UwbOnlineInitialisation object
        drone_positions = self.uwb_online_initialisation.unknown_anchor_measurements[anchor_id]["positions_pre_rough_estimate"]
        range_measurements = self.uwb_online_initialisation.unknown_anchor_measurements[anchor_id]["distances_pre_rough_estimate"]

        # Run the pipleline
        for i in range(len(drone_positions)):

            # Append the measurement to the measurement vector
            self.measurement_vector.append([drone_positions[i][0], drone_positions[i][1], drone_positions[i][2], range_measurements[i]])

            # Check if there are enough measurements to run the linear least squares estimation and compute metrics
            if len(self.measurement_vector) > 5:

                # Run the linear least squares estimation to compute the estimate, the covariance matrix and the residuals
                estimator, covariance_matrix, residuals, _ = self.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(self.measurement_vector)

                # Compute the stopping criterion variables
                FIM = self.uwb_online_initialisation.compute_FIM(self.measurement_vector, estimator[:3])
                GDOP = self.uwb_online_initialisation.compute_GDOP(self.measurement_vector, estimator[:3])
                sum_of_residuals = np.mean(residuals)
                condition_number = self.uwb_online_initialisation.compute_condition_number(self.measurement_vector)
                covariances = np.diag(covariance_matrix)

                error = self.uwb_online_initialisation.calculate_position_error(0, estimator[:3])

                self.gdop_vector.append(GDOP)
                self.fim_vector.append(np.linalg.det(FIM))
                self.condition_number_vector.append(condition_number)
                self.covariances_vector.append(covariances)
                self.sum_of_residuals_vector.append(sum_of_residuals)
                self.error_vector.append(error)

                # Find outliers in the residuals and remove them from the measurement vector
                outliers = self.uwb_online_initialisation.outlier_finder(residuals)
                for outlier in outliers[::-1]:
                    self.measurement_vector.pop(outlier)

            else: # If there are not enough measurements, append NaN values to the vectors
                self.gdop_vector.append(np.nan)
                self.fim_vector.append(np.nan)
                self.condition_number_vector.append(np.nan)
                self.covariances_vector.append([np.nan, np.nan, np.nan])
                self.sum_of_residuals_vector.append(np.nan)
                self.error_vector.append(np.nan)

    



    def run_simulation(self):
        """Run a simulation by gathering measurements and metrics for the unknown anchor, using all space and all measurement available. (no stopping criterion)"""

        self.uwb_online_initialisation.process_anchor_info(self.drone_sim.base_anchors, self.drone_sim.unknown_anchors)

        unknown_anchor = self.drone_sim.unknown_anchors[0]
        self.reset_metrics(unknown_anchor.anchor_ID)

        self.gather_measurements() # Gather measurements (keep them stored in the UwbOnlineInitialisation object)
        self.gather_metrics(unknown_anchor.anchor_ID) # Gather metrics

    def run_simulation_ground_truth(self):
        """Run a simulation by gathering measurements and metrics for the unknown anchor, using all space and all measurement available. (no stopping criterion)
        The measurements are gathered using the ground truth distance (no bias and no noise)."""

        self.uwb_online_initialisation.process_anchor_info(self.drone_sim.base_anchors, self.drone_sim.unknown_anchors)

        unknown_anchor = self.drone_sim.unknown_anchors[0]
        self.reset_metrics(unknown_anchor.anchor_ID)

        self.gather_measurements() # Gather measurements (keep them stored in the UwbOnlineInitialisation object)
        self.gather_metrics(unknown_anchor.anchor_ID) # Gather metrics

    def run_simulation_outlier_filtering(self):
        """Run a simulation by gathering measurements and metrics for the unknown anchor, using all space and all measurement available. (no stopping criterion)
        The measurements are gathered by simultaneously removing outliers from the measurements."""

        self.uwb_online_initialisation.process_anchor_info(self.drone_sim.base_anchors, self.drone_sim.unknown_anchors)

        unknown_anchor = self.drone_sim.unknown_anchors[0]
        self.reset_metrics(unknown_anchor.anchor_ID)

        self.gather_measurements() # Gather measurements (keep them stored in the UwbOnlineInitialisation object)
        self.gather_metrics_with_outlier_filtering(unknown_anchor.anchor_ID) # Gather metrics

    def use_stopping_criterion(self, metric='gdop', threshold=2):
        """Function to use a stopping criterion to extract a subset of the measurements.

        Parameters:
        metric (str): The metric to use for the stopping criterion. Can be 'gdop', 'fim', 'condition_number', 'covariance', or 'sum_of_residuals'.
        threshold (float): The threshold value for the stopping criterion.
        
        Returns:
        measurement_vector (list): The subset of measurements that satisfy the stopping criterion.


        Good values for the threshold:
        - gdop: 2
        - fim: 2
        - condition_number: 500
        - covariance: 0.1
        - sum_of_residuals: 1

        """

        if metric == 'gdop':
            # Extract measurement vector until the corresponding gdop is below a certain threshold = 2
            gdop_index = np.where(np.array(self.gdop_vector) < threshold)[0][0]
            measurement_vector = self.measurement_vector[:gdop_index]

        elif metric == 'fim':
            # Extract measurement vector until the corresponding fim is below a certainn threshold = 2
            fim_index = np.where(1/np.linalg.det(np.array(self.fim_vector)) < threshold)[0][0]
            measurement_vector = self.measurement_vector[:fim_index]

        
        elif metric == 'condition_number':
            # Extract measurement vector until the corresponding condition number is below a certain threshold = 2
            condition_number_index = np.where(np.array(self.condition_number_vector) < threshold)[0][0]
            measurement_vector = self.measurement_vector[:condition_number_index]

        elif metric == 'covariance':
            # Extract measurement vector until the corresponding covariance is below a certain threshold = 0.1
            max_values = [np.min(triplet[:3]) for triplet in self.covariances_vector]

            # Convert max values to a NumPy array for easy indexing
            max_values_array = np.array(max_values)
            # Find indices where the maximum value is below 0.1
            cov_index = np.where(max_values_array < threshold)[0][0]
            measurement_vector = self.measurement_vector[:cov_index]

        elif metric == 'sum_of_residuals':
            # Extract measurement vector until the corresponding sum of residuals is below a certain threshold = 0.1
            sum_of_residuals_index = np.where(np.array(self.sum_of_residuals_vector) < threshold)[0][0]
            measurement_vector = self.measurement_vector[:sum_of_residuals_index]

        elif metric == None:
            measurement_vector = self.measurement_vector
        
        return measurement_vector



        

    def write_measurements_to_csv(self, path):
        """Function to create some data with measurements and write it to a CSV file."""
        # Open a CSV file in append mode
        anchor_id = self.drone_sim.unknown_anchors[0].anchor_ID

        row = [str(self.drone_sim.unknown_anchors[0].get_anchor_coordinates()), str(self.uwb_online_initialisation.unknown_anchor_measurements[anchor_id]["positions_pre_rough_estimate"]), str(self.uwb_online_initialisation.unknown_anchor_measurements[anchor_id]["distances_pre_rough_estimate"])]
        
        # Open a CSV file in append mode
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def load_measurement_data_from_csv(self, path):

        def convert_str_to_list(s):
            # Replace 'nan', 'inf', and '-inf' with their corresponding numpy constants
            s = s.replace('nan', 'np.nan').replace('inf', 'np.inf').replace('-inf', '-np.inf')
            # Evaluate the string as a Python expression and return the result
            try:
                return eval(s)
            except Exception as e:
                # If evaluation fails, return the original string
                return s
    
        data = pd.read_csv(path, header=None, converters={i: convert_str_to_list for i in range(3)})
        data.columns = ['anchor_position_gt','measured_positions', 'measured_ranges']

        return data




    def compute_angular_span_around_point(self, vectors, point):
        """Utility function to compute the angular span around a given point.
        
        Parameters:
        vectors (list): List of vectors in the form (x, y, z).
        point (list): The point around which to compute the angular span in the form (x0, y0, z0).
        
        Returns:
        azimuthal_span (float): The azimuthal span around the point.
        elevation_span (float): The elevation span around the point.
        """

        def cartesian_to_spherical(vector, point):
            # Translate the vector so that 'point' is the new origin
            translated_vector = (vector[0] - point[0], vector[1] - point[1], vector[2] - point[2])
            x, y, z = translated_vector
            
            # Compute spherical coordinates
            r = np.sqrt(x**2 + y**2 + z**2)  # Radial distance (not used directly here)
            azimuthal_angle = np.arctan2(y, x)  # Azimuthal angle (phi)
            elevation_angle = np.arctan2(z, np.sqrt(x**2 + y**2))  # Elevation angle (theta)
            
            return azimuthal_angle, elevation_angle
        
        def angular_differences(angles):
            # Compute differences between consecutive angles
            diffs = np.diff(angles, append=angles[0] + (angles[-1] - angles[0]))  # Wrap around
            diffs = np.where(diffs > np.pi, diffs - 2 * np.pi, diffs)  # Handle positive wrap-around
            diffs = np.where(diffs < -np.pi, diffs + 2 * np.pi, diffs)  # Handle negative wrap-around
            return diffs
        
        def kadane_algorithm(arr):
            # Find maximum sum subarray (positive span)
            max_ending_here = max_so_far = arr[0]
            for x in arr[1:]:
                max_ending_here = max(x, max_ending_here + x)
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        def min_kadane_algorithm(arr):
            # Find minimum sum subarray (negative span)
            min_ending_here = min_so_far = arr[0]
            for x in arr[1:]:
                min_ending_here = min(x, min_ending_here + x)
                min_so_far = min(min_so_far, min_ending_here)
            return min_so_far

        # Convert vectors to spherical coordinates
        spherical_coords = [cartesian_to_spherical(vector, point) for vector in vectors]
        
        azimuthal_angles = [angle[0] for angle in spherical_coords]
        elevation_angles = [angle[1] for angle in spherical_coords]
        
        # Compute angular differences for azimuthal and elevation angles
        azimuthal_diffs = angular_differences(azimuthal_angles)
        elevation_diffs = angular_differences(elevation_angles)
        
        # Compute the maximum and minimum angular spans using Kadane’s algorithms
        max_azimuthal_span = kadane_algorithm(azimuthal_diffs)
        min_azimuthal_span = min_kadane_algorithm(azimuthal_diffs)
        max_elevation_span = kadane_algorithm(elevation_diffs)
        min_elevation_span = min_kadane_algorithm(elevation_diffs)
        
        # The largest span is the maximum of absolute values of max and min spans
        azimuthal_span = max(abs(max_azimuthal_span), abs(min_azimuthal_span))
        elevation_span = max(abs(max_elevation_span), abs(min_elevation_span))
        
        azimuthal_span_degrees = azimuthal_span * 180 / np.pi
        elevation_span_degrees = elevation_span * 180 / np.pi

        return azimuthal_span_degrees, elevation_span_degrees


    def run_residual_analysis(self):

        self.randomise_environment()
        self.uwb_online_initialisation.process_anchor_info(self.drone_sim.base_anchors, self.drone_sim.unknown_anchors)

        unknown_anchor = self.drone_sim.unknown_anchors[0]

        self.reset_metrics(unknown_anchor.anchor_ID)

        self.gather_measurements()
        anchor_id = unknown_anchor.anchor_ID
        drone_positions = self.uwb_online_initialisation.unknown_anchor_measurements[anchor_id]["positions_pre_rough_estimate"]
        range_measurements = self.uwb_online_initialisation.unknown_anchor_measurements[anchor_id]["distances_pre_rough_estimate"]

        residual_vector = []
        for i in range(len(drone_positions)):
            self.measurement_vector.append([drone_positions[i][0], drone_positions[i][1], drone_positions[i][2], range_measurements[i]])

            if len(self.measurement_vector) > 5:
                # Run the linear least squares estimation and compute the residuals
                estimator, covariance_matrix, residuals, x = self.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(self.measurement_vector)

                residual_vector.append(residuals)




        self.print_high_residual_measurements(np.array(drone_positions), residual_vector[-1], range_measurements, unknown_anchor.get_anchor_coordinates(), x)

        
        #self.plot_colored_residuals(unknown_anchor.get_anchor_coordinates(), np.array(drone_positions), residual_vector[-1], estimator[:3])





    def plot_colored_residuals(self, ground_truth, measurements, residuals, estimated_position):
        # Normalize residuals for color mapping
        norm = plt.Normalize(residuals.min(), residuals.max())

        # Create a figure with two subplots: 3D scatter plot and residuals plot
        fig = plt.figure(figsize=(12, 6))

        # Subplot 1: 3D Scatter Plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(ground_truth[0], ground_truth[1], ground_truth[2], c='r', marker='o', s=100, label='Ground Truth')
        
        scatter = ax1.scatter(measurements[:, 0], measurements[:, 1], measurements[:, 2], 
                            c=residuals, cmap='coolwarm', norm=norm, s=50)  # Changed colormap to coolwarm

        # Add color bar to show the scale of residuals
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.5, aspect=5)
        cbar.set_label('Residual Magnitude')

        # Plot estimated position
        ax1.scatter(estimated_position[0], estimated_position[1], estimated_position[2], c='g', marker='^', s=100, label='Estimated Position')

        # Calculate the error between the estimated position and the ground truth
        error = np.linalg.norm(estimated_position - ground_truth)

        # Add text for the error
        ax1.text(estimated_position[0], estimated_position[1], estimated_position[2], 
                f'Error: {error:.2f} m', color='black', fontsize=12)

        # Set plot labels
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Scatter Plot with Ground Truth, Measurements, and Estimated Position')

        # Show legend
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=[handles[0], handles[1]])  # Include both Ground Truth and Estimated Position

        # Subplot 2: Residuals Plot
        ax2 = fig.add_subplot(122)
        indices = np.arange(len(residuals))
        ax2.scatter(indices, residuals, c='b', marker='o')
        ax2.set_xlabel('Measurement Index')
        ax2.set_ylabel('Residual Magnitude')
        ax2.set_title('Residuals as a Function of Measurement Index')

        # Show plot
        plt.tight_layout()
        plt.show()

        # Print the error
        print(f'Error between estimated position and ground truth: {error:.2f} meters')

    def print_high_residual_measurements(self, measurements, residuals, ranges, ground_truth, x_ls, threshold=100):
        """
        Print measurements with residuals above the given threshold, including their distances to the ground truth
        and the delta between the range and ground truth distance.
        """
        high_residual_indices = np.where(residuals > threshold)[0]
        
        range_measurements = np.array(ranges)
        measurements = np.array(measurements)

        # Ensure measurements and range are 2D arrays
        range_measurements = range_measurements.reshape(-1, 1)  # Ensure it's a column vector

        # Combine measurements and range
        combined_matrix = np.hstack((measurements, range_measurements))
        A, b = self.uwb_online_initialisation.setup_linear_least_square(combined_matrix)

        print("Result of the linear LS", x_ls)
        print("Ground truth", ground_truth, np.linalg.norm(ground_truth)**2)
        self.plot_colored_residuals(ground_truth, np.array(combined_matrix), residuals, x_ls[:3])
        if len(high_residual_indices) == 0:
            print("No measurements with residuals above the threshold.")
        else:
            print(f"Measurements with residuals above {threshold}:")
            for index in high_residual_indices:
                measurement_position = measurements[index]
                # Calculate the Euclidean distance from the measurement position to the ground truth
                distance_to_ground_truth = np.linalg.norm(measurement_position - ground_truth)
                # Calculate the delta between the measured range and the distance to ground truth
                delta = np.abs(ranges[index] - distance_to_ground_truth)

                computed_residual = -(A[index] @ x_ls - b[index])

                x, y, z = measurement_position
                measured_dist = ranges[index]
                norm_squared = x**2 + y**2 + z**2
                a = [-2*x, -2*y, -2*z, 1]
                B = measured_dist**2 - norm_squared


                print(f"Measurement {index}: Position {measurement_position}, Range {ranges[index]}, Residual {residuals[index]:.2f}, True residual {computed_residual:.2f} m, Distance to Ground Truth {distance_to_ground_truth:.2f} m, Delta {delta:.2f} m")

        # Exclude high residual measurements
        filtered_indices = np.setdiff1d(np.arange(len(measurements)), high_residual_indices)
        filtered_measurements = measurements[filtered_indices]
        filtered_ranges = range_measurements[filtered_indices]

        # Recompute least squares estimation without high residuals
        combined_filtered_matrix = np.hstack((filtered_measurements, filtered_ranges.reshape(-1, 1)))
        A_filtered, b_filtered = self.uwb_online_initialisation.setup_linear_least_square(combined_filtered_matrix)
        estimator_filtered, covariance_matrix_filtered, residuals_filtered, x_filtered = self.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(combined_filtered_matrix)

        # Print new results
        print("Result of the linear LS without high residuals", estimator_filtered)
        print("Ground truth", ground_truth, np.linalg.norm(ground_truth)**2)
        
        # Print residuals for the filtered results
        self.plot_colored_residuals(ground_truth, np.array(filtered_measurements), residuals_filtered, estimator_filtered[:3])
        print("New residual analysis:")
        for index in range(len(filtered_measurements)):
            measurement_position = filtered_measurements[index]
            distance_to_ground_truth = np.linalg.norm(measurement_position - ground_truth)
            delta = np.abs(filtered_ranges[index] - distance_to_ground_truth)

            computed_residual = -(A_filtered[index] @ x_filtered - b_filtered[index])

            x, y, z = measurement_position
            measured_dist = filtered_ranges[index]
            norm_squared = x**2 + y**2 + z**2
            a = [-2*x, -2*y, -2*z, 1]
            B = measured_dist**2 - norm_squared

            print(f"Measurement {index}: Position {measurement_position}, Range {filtered_ranges[index]}, Residual {residuals_filtered[index]:.2f}, True residual {computed_residual:.2f} m, Distance to Ground Truth {distance_to_ground_truth:.2f} m, Delta {delta.tolist()[0]:.2f} m")


    def use_ransac(self, measurements, num_iterations=1000, threshold=0.5):

        def trilaterate(p1, p2, p3, p4, r1, r2, r3, r4):
            """
            Compute the position using trilateration given 4 points and their distances.
            Points are in the form (x, y, z) and distances are the respective measurements.
            """
            # Unpack the points
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            x3, y3, z3 = p3
            x4, y4, z4 = p4
            
            # Convert distances to squared values
            r1_squared = r1 ** 2
            r2_squared = r2 ** 2
            r3_squared = r3 ** 2
            r4_squared = r4 ** 2
            
            # Set up matrices for solving the system of equations
            A = np.array([
                [2 * (x2 - x1), 2 * (y2 - y1), 2 * (z2 - z1)],
                [2 * (x3 - x1), 2 * (y3 - y1), 2 * (z3 - z1)],
                [2 * (x4 - x1), 2 * (y4 - y1), 2 * (z4 - z1)]
            ])
            
            b = np.array([
                r1_squared - r2_squared - (x1**2 - x2**2) - (y1**2 - y2**2) - (z1**2 - z2**2),
                r1_squared - r3_squared - (x1**2 - x3**2) - (y1**2 - y3**2) - (z1**2 - z3**2),
                r1_squared - r4_squared - (x1**2 - x4**2) - (y1**2 - y4**2) - (z1**2 - z4**2)
            ])
            
            # Solve for the position
            try:
                position = np.linalg.lstsq(A, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None  # In case of an ill-conditioned matrix
            
            return position

        def calculate_residuals(position, measurements):
            """
            Compute residuals for each measurement given the estimated position.
            """
            residuals = []
            for (x, y, z, r) in measurements:
                dist = np.linalg.norm(np.array(position) - np.array([x, y, z]))
                residual = abs(dist - r)
                residuals.append(residual)
            return residuals

        best_model = None
        best_inliers = []
        
        for _ in range(num_iterations):
            # Randomly sample 4 measurements
            sample = random.sample(measurements, 4)
            points, distances = zip(*[(m[:3], m[3]) for m in sample])
            
            # Compute the estimated position
            estimated_position = trilaterate(*points, *distances)
            
            if estimated_position is None:
                continue
            
            # Calculate residuals
            residuals = calculate_residuals(estimated_position, measurements)
            
            # Determine inliers
            inliers = [i for i, res in enumerate(residuals) if res < threshold]
            
            # Update best model if current one has more inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = estimated_position
        
        return best_model, best_inliers

















def run_simulation_metrics():
    """Run the simulation to gather measurements and metrics for the unknown anchor, using all space and all measurement available. (no stopping criterion)
    The metrics are stored in a CSV file, ready to be analysed.""" 

    # Define the path of the csv to store the data
    path = csv_dir / 'metrics.csv'

    # iterate over environments
    for environment in range(1000):

        calculate_online_initialisation = CalculateOnlineInitialisation()

        # Randomise the environment
        calculate_online_initialisation.randomise_environment()

        # Run the simulation to gather measurements and metrics without stopping criterion
        calculate_online_initialisation.run_simulation_outlier_filtering()
        
        row = [str(calculate_online_initialisation.gdop_vector), str(calculate_online_initialisation.fim_vector), str(calculate_online_initialisation.condition_number_vector), str(calculate_online_initialisation.sum_of_residuals_vector), str([np.max(row) for row in calculate_online_initialisation.covariances_vector]), str(calculate_online_initialisation.error_vector)]

        # Open a CSV file in append mode
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

def run_simulation_linear_least_squares_comparison():
    """Run the simulation to gather measurements and compute a linear least squares estimation for the unknown anchor, using different bias settings.
    A stopping criterion is used to extract a subset of the measurements, and the results are stored in a CSV file."""
    
    # Define the path of the csv to store the data
    path = csv_dir / 'linear_least_squares.csv'

    # Iterate over environments
    for environment in range(100000):

        calculate_online_initialisation = CalculateOnlineInitialisation()

        # Randomise the environment
        calculate_online_initialisation.randomise_environment()

        # Run the simulation to gather measurements and metrics without stopping criterion
        calculate_online_initialisation.run_simulation_outlier_filtering()

        # Use the stopping criterion to extract a subset of the measurements
        try:
            measurement_vector = calculate_online_initialisation.use_stopping_criterion()
        except:
            continue

        if len(measurement_vector) < 5:
            continue
        
        # Run the linear least squares estimation with and without biases and store them in arrays
        estimators = []
        covariances = []
        residualss = []

        for use_linear_bias in [True, False]:
            for use_constant_bias in [True, False]:
                calculate_online_initialisation.uwb_online_initialisation.params['use_linear_bias'] = use_linear_bias
                calculate_online_initialisation.uwb_online_initialisation.params['use_constant_bias'] = use_constant_bias

                estimator, covariance, residuals, _ = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(measurement_vector)
                
                estimators.append(estimator)
                covariances.append(covariance)
                residualss.append(residuals)

        # position, inlier_indices = calculate_online_initialisation.use_ransac(measurement_vector, num_iterations=1000, threshold=0.5)

        # Calculate the error values
        error_t_t = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimators[0][:3])
        error_t_t_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[0])
        error_t_f = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[1][:3])
        error_t_f_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[1])
        error_f_t = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[2][:3])
        error_f_t_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[2])
        error_f_f = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[3][:3])
        error_f_f_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[3])

        # error_position_ransac = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), position)

        # Calculate metrics to store in the CSV
        number_of_measurements = len(measurement_vector)
        min_angle_between_two_consecutive_measurements = np.rad2deg(np.atan(calculate_online_initialisation.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold']))

        anchor_position_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
        anchor_distances_gt = np.linalg.norm([np.array(row[:3]) - np.array(anchor_position_gt) for row in measurement_vector] , axis=1)

        measured_noise_mean = np.mean(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))
        measured_noise_var = np.var(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))

        min_distance_to_anchor = np.min(anchor_distances_gt)
        max_distance_to_anchor = np.max(anchor_distances_gt)
        mean_distance_to_anchor = np.mean(anchor_distances_gt)
        std_distance_to_anchor = np.std(anchor_distances_gt)

        angular_span_azimuth, angular_span_elevation = calculate_online_initialisation.compute_angular_span_around_point([row[:3] for row in measurement_vector], anchor_position_gt)

        constant_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].bias
        linear_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias

        linear_bias_error_true_true = np.linalg.norm(estimators[0][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_bias_error_true_false = np.linalg.norm(estimators[1][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)

        constant_bias_error_true_true = np.linalg.norm(estimators[0][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)
        constant_bias_error_false_true = np.linalg.norm(estimators[2][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)

        noise_variance = calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance
        outlier_probability = calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability

        row = [number_of_measurements, min_angle_between_two_consecutive_measurements, min_distance_to_anchor, max_distance_to_anchor, mean_distance_to_anchor, std_distance_to_anchor, angular_span_elevation, angular_span_azimuth, constant_bias_gt, linear_bias_gt, measured_noise_mean, noise_variance, measured_noise_var, outlier_probability, error_t_t, error_t_t_full, constant_bias_error_true_true, linear_bias_error_true_true, error_t_f, error_t_f_full, linear_bias_error_true_false, error_f_t, error_f_t_full, constant_bias_error_false_true, error_f_f, error_f_f_full]


        # Open a CSV file in append mode
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

def run_simulation_non_linear_least_squares_comparison():
    """Run the simulation to gather measurements and compute a non-linear least squares estimation for the unknown anchor, using different optimisation methods.
    A stopping criterion is used to extract a subset of the measurements, and the results are stored in a CSV file."""

    # Define the path of the csv to store the data
    path = csv_dir / 'non_linear_least_squares.csv'

    # Iterate over environments
    for environment in range(100000):

        calculate_online_initialisation = CalculateOnlineInitialisation()

        # Randomise the environment
        calculate_online_initialisation.randomise_environment()

        # Run the simulation to gather measurements and metrics without stopping criterion
        calculate_online_initialisation.run_simulation()

        # Use the stopping criterion to extract a subset of the measurements
        try:
            measurement_vector = calculate_online_initialisation.use_stopping_criterion()
        except:
            continue
        
        if len(measurement_vector) < 5:
            continue
        
        # Run the linear least squares estimation
        calculate_online_initialisation.uwb_online_initialisation.params['use_linear_bias'] = False
        calculate_online_initialisation.uwb_online_initialisation.params['use_constant_bias'] = False
        estimator_linear, covariance_linear, residuals_linear, _ = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(measurement_vector)

        # Run the non-linear least squares estimation with different optimisation methods
        non_linear_methods = ['LM', 'IRLS', 'KRR']
        estimators = []
        covariances = []
        residualss = []

        for method in non_linear_methods:
            calculate_online_initialisation.uwb_online_initialisation.params['non_linear_optimisation_type'] = method
            estimator, covariance = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_non_linear_least_squares(measurement_vector, estimator_linear)

            estimators.append(estimator)
            covariances.append(covariance)


        # Calculate the error values
        linear_error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimator_linear[:3])
        linear_error_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimator_linear)
        linear_linear_bias_error = np.linalg.norm(estimator_linear[4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_constant_bias_error = np.linalg.norm(estimator_linear[3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)


        error_lm = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimators[0][:3])
        error_lm_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[0])

        error_irls = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[1][:3])
        error_irls_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[1])

        error_krr = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[2][:3])
        error_krr_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[2])

        # Calculate metrics to store in the CSV
        number_of_measurements = len(measurement_vector)

        min_angle_between_two_consecutive_measurements = np.rad2deg(np.atan(calculate_online_initialisation.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold']))

        anchor_position_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
        anchor_distances_gt = np.linalg.norm([np.array(row[:3]) - np.array(anchor_position_gt) for row in measurement_vector] , axis=1)

        measured_noise_mean = np.mean(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))
        measured_noise_var = np.var(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))

        min_distance_to_anchor = np.min(anchor_distances_gt)
        max_distance_to_anchor = np.max(anchor_distances_gt)
        mean_distance_to_anchor = np.mean(anchor_distances_gt)
        std_distance_to_anchor = np.std(anchor_distances_gt)

        angular_span_azimuth, angular_span_elevation = calculate_online_initialisation.compute_angular_span_around_point([row[:3] for row in measurement_vector], anchor_position_gt)

        constant_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].bias
        linear_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias

        linear_bias_error_lm = np.linalg.norm(estimators[0][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_bias_error_irls = np.linalg.norm(estimators[1][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_bias_error_krr = np.linalg.norm(estimators[2][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        
        
        constant_bias_error_lm = np.linalg.norm(estimators[0][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)
        constant_bias_error_irls = np.linalg.norm(estimators[1][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)
        constant_bias_error_krr = np.linalg.norm(estimators[2][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)

        noise_variance = calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance
        outlier_probability = calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability

        row = [number_of_measurements, min_angle_between_two_consecutive_measurements, min_distance_to_anchor, max_distance_to_anchor, mean_distance_to_anchor, std_distance_to_anchor, angular_span_elevation, angular_span_azimuth, constant_bias_gt, linear_bias_gt, measured_noise_mean, noise_variance, measured_noise_var, outlier_probability, linear_error, linear_error_full, linear_linear_bias_error, linear_constant_bias_error, error_lm, error_lm_full, constant_bias_error_lm, linear_bias_error_lm, error_irls, error_irls_full, constant_bias_error_irls ,linear_bias_error_irls, error_krr, error_krr_full, constant_bias_error_krr, linear_bias_error_krr]


        # Open a CSV file in append mode
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

def run_simulation_outlier_filtering_comparison():
    """Run the simulation to gather measurements and compute a non-linear least squares estimation for the unknown anchor, using different outlier filtering methods.
    A stopping criterion is used to extract a subset of the measurements, and the results are stored in a CSV file."""

    # Define the path of the csv to store the data
    path = csv_dir / 'outlier_filtering.csv'

    # Iterate over environments
    for environment in range(100000):

        calculate_online_initialisation = CalculateOnlineInitialisation()

        # Randomise the environment
        calculate_online_initialisation.randomise_environment()

        # Run the simulation to gather measurements and metrics without stopping criterion
        calculate_online_initialisation.run_simulation()

        try:
            measurement_vector = calculate_online_initialisation.use_stopping_criterion(metric=None)
        except:
            continue

        if len(measurement_vector) < 5:
            continue

        # Run the simulation to gather measurements and metrics without stopping criterion but with outlier filtering
        calculate_online_initialisation.run_simulation_outlier_filtering()
        try:
            measurement_vector_outlier_filtering = calculate_online_initialisation.use_stopping_criterion()
        except:
            continue

        if len(measurement_vector_outlier_filtering) < 5:
            continue
        
        # Run the linear least squares estimation with and without outlier filtering
        estimators = []
        covariances = []
        residualss = []

        calculate_online_initialisation.uwb_online_initialisation.params['use_linear_bias'] = False
        calculate_online_initialisation.uwb_online_initialisation.params['use_constant_bias'] = False
        estimator_linear, covariance_linear, residuals_linear, _ = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(measurement_vector)
        estimator_linear_outlier_filtering, covariance_linear_outlier_filtering, residuals_linear_outlier_filtering, _ = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(measurement_vector_outlier_filtering)
        
        calculate_online_initialisation.uwb_online_initialisation.params['non_linear_optimisation_type'] = 'IRLS'
        estimator, covariance = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_non_linear_least_squares(measurement_vector, estimator_linear)
        estimator_outlier_filtering, covariance_outlier_filtering = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_non_linear_least_squares(measurement_vector_outlier_filtering, estimator_linear_outlier_filtering)

        estimators.append(estimator)
        covariances.append(covariance)
        estimators.append(estimator_outlier_filtering)
        covariances.append(covariance_outlier_filtering)


        # Calculate the error values
        linear_error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimator_linear[:3])
        linear_error_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimator_linear)
        linear_linear_bias_error = np.linalg.norm(estimator_linear[4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_constant_bias_error = np.linalg.norm(estimator_linear[3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)

        linear_error_outlier_filtering = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimator_linear_outlier_filtering[:3])
        linear_error_full_outlier_filtering = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimator_linear_outlier_filtering)
        linear_linear_bias_error_outlier_filtering = np.linalg.norm(estimator_linear_outlier_filtering[4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_constant_bias_error_outlier_filtering = np.linalg.norm(estimator_linear_outlier_filtering[3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)


        error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimators[0][:3])
        error_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[0])
        error_outlier_filtering = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[1][:3])
        error_full_outlier_filtering = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[1])


        # Calculate metrics to store in the CSV
        number_of_measurements = len(measurement_vector)
        number_of_measurements_outlier_filtering = len(measurement_vector_outlier_filtering)
        
        min_angle_between_two_consecutive_measurements = np.rad2deg(np.atan(calculate_online_initialisation.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold']))

        anchor_position_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
        anchor_distances_gt = np.linalg.norm([np.array(row[:3]) - np.array(anchor_position_gt) for row in measurement_vector] , axis=1)

        measured_noise_mean = np.mean(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))
        measured_noise_var = np.var(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))

        min_distance_to_anchor = np.min(anchor_distances_gt)
        max_distance_to_anchor = np.max(anchor_distances_gt)
        mean_distance_to_anchor = np.mean(anchor_distances_gt)
        std_distance_to_anchor = np.std(anchor_distances_gt)

        angular_span_azimuth, angular_span_elevation = calculate_online_initialisation.compute_angular_span_around_point([row[:3] for row in measurement_vector], anchor_position_gt)

        constant_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].bias
        linear_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias

        linear_bias_error = np.linalg.norm(estimators[0][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_bias_error_outlier_filtering = np.linalg.norm(estimators[1][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        
        
        constant_bias_error = np.linalg.norm(estimators[0][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)
        constant_bias_error_outlier_filtering = np.linalg.norm(estimators[1][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)

        noise_variance = calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance
        outlier_probability = calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability

        row = [number_of_measurements, number_of_measurements_outlier_filtering, min_angle_between_two_consecutive_measurements, min_distance_to_anchor, max_distance_to_anchor, mean_distance_to_anchor, std_distance_to_anchor, angular_span_elevation, angular_span_azimuth, constant_bias_gt, linear_bias_gt, measured_noise_mean, noise_variance, measured_noise_var, outlier_probability, linear_error, linear_error_full, linear_linear_bias_error, linear_constant_bias_error, linear_error_outlier_filtering, linear_error_full_outlier_filtering, linear_linear_bias_error_outlier_filtering, linear_constant_bias_error_outlier_filtering, error, error_full, constant_bias_error, linear_bias_error, error_outlier_filtering, error_full_outlier_filtering, constant_bias_error_outlier_filtering ,linear_bias_error_outlier_filtering]

        # Open a CSV file in append mode
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)



                
            



def run_simulation_trajectory_optimisation_comparison():
    pass

def run_simulation_residual_analysis():
    calculate_online_initialisation = CalculateOnlineInitialisation()
    calculate_online_initialisation.run_residual_analysis()

def run_simulation_error_debugging():
    path = csv_dir / 'error_debugging.csv'

    for environment in range(100000):
        calculate_online_initialisation = CalculateOnlineInitialisation()

        calculate_online_initialisation.randomise_environment()

        calculate_online_initialisation.run_simulation()
        # measurement_vector = calculate_online_initialisation.use_stopping_criterion()
        try:
            measurement_vector = calculate_online_initialisation.use_stopping_criterion()
        except:
            continue

        if len(measurement_vector) < 5:
            continue
        
        num_measurements = len(measurement_vector)

        position, inlier_indices = calculate_online_initialisation.use_ransac(measurement_vector, num_iterations=1000, threshold=0.2)

        if position is None:
            continue

        calculate_online_initialisation.uwb_online_initialisation.params['use_linear_bias'] = False
        calculate_online_initialisation.uwb_online_initialisation.params['use_constant_bias'] = False

        estimator, covariance, residuals, _ = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(measurement_vector)
        
        error_position_ransac = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), position)
        error_position_ls = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimator[:3])

        anchor_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator()

        row = [num_measurements, anchor_gt[0], anchor_gt[1], anchor_gt[2], position[0], position[1], position[2], error_position_ransac, estimator[0], estimator[1], estimator[2], error_position_ls]
        
        # Open a CSV file in append mode
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

def run_simulation_clustering_comparison():

    path = csv_dir / 'clustering.csv'

    for environment in range(100000):
        calculate_online_initialisation = CalculateOnlineInitialisation()

        calculate_online_initialisation.randomise_environment()

        calculate_online_initialisation.run_simulation()
        try:
            measurement_vector = calculate_online_initialisation.use_stopping_criterion()
        except:
            continue

        if len(measurement_vector) < 5:
            continue

        estimators = []
        covariances = []
        residualss = []

        calculate_online_initialisation.uwb_online_initialisation.params['use_linear_bias'] = False
        calculate_online_initialisation.uwb_online_initialisation.params['use_constant_bias'] = False
        estimator_linear, covariance_linear, residuals_linear, _ = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_linear_least_squares(measurement_vector)

        
        calculate_online_initialisation.uwb_online_initialisation.params['non_linear_optimisation_type'] = 'IRLS'
        estimator, covariance = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_non_linear_least_squares(measurement_vector, estimator_linear)

        estimators.append(estimator)
        covariances.append(covariance)

        outliers = calculate_online_initialisation.uwb_online_initialisation.outlier_finder(residuals_linear)

        for outlier in outliers[::-1]:
            measurement_vector.pop(outlier)

        clustered_measurements = calculate_online_initialisation.uwb_online_initialisation.cluster_measurements(measurement_vector, 15)
        estimator, covariance = calculate_online_initialisation.uwb_online_initialisation.estimate_anchor_position_non_linear_least_squares(clustered_measurements, estimator_linear)

        estimators.append(estimator)
        covariances.append(covariance)    


        # Save error values to CSV

        linear_error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimator_linear[:3])
        linear_error_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimator_linear)
        linear_linear_bias_error = np.linalg.norm(estimator_linear[4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_constant_bias_error = np.linalg.norm(estimator_linear[3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)


        error = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates(), estimators[0][:3])
        error_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[0])

        error_cluster = calculate_online_initialisation.uwb_online_initialisation.calculate_position_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[1][:3])
        error_cluster_full = calculate_online_initialisation.uwb_online_initialisation.calculate_estimator_error(calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_gt_estimator(), estimators[1])


        number_of_measurements = len(measurement_vector)

        min_angle_between_two_consecutive_measurements = np.rad2deg(np.atan(calculate_online_initialisation.uwb_online_initialisation.params['distance_to_anchor_ratio_threshold']))

        anchor_position_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].get_anchor_coordinates()
        anchor_distances_gt = np.linalg.norm([np.array(row[:3]) - np.array(anchor_position_gt) for row in measurement_vector] , axis=1)

        measured_noise_mean = np.mean(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))
        measured_noise_var = np.var(np.abs(anchor_distances_gt - np.array(measurement_vector)[:, 3]))

        min_distance_to_anchor = np.min(anchor_distances_gt)
        max_distance_to_anchor = np.max(anchor_distances_gt)
        mean_distance_to_anchor = np.mean(anchor_distances_gt)
        std_distance_to_anchor = np.std(anchor_distances_gt)

        angular_span_azimuth, angular_span_elevation = calculate_online_initialisation.compute_angular_span_around_point([row[:3] for row in measurement_vector], anchor_position_gt)

        constant_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].bias
        linear_bias_gt = calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias

        linear_bias_error = np.linalg.norm(estimators[0][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        linear_bias_error_cluster = np.linalg.norm(estimators[1][4] - calculate_online_initialisation.drone_sim.unknown_anchors[0].linear_bias)
        
        
        constant_bias_error = np.linalg.norm(estimators[0][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)
        constant_bias_error_cluster = np.linalg.norm(estimators[1][3] - calculate_online_initialisation.drone_sim.unknown_anchors[0].bias)

        noise_variance = calculate_online_initialisation.drone_sim.unknown_anchors[0].noise_variance
        outlier_probability = calculate_online_initialisation.drone_sim.unknown_anchors[0].outlier_probability

        row = [number_of_measurements, min_angle_between_two_consecutive_measurements, min_distance_to_anchor, max_distance_to_anchor, mean_distance_to_anchor, std_distance_to_anchor, angular_span_elevation, angular_span_azimuth, constant_bias_gt, linear_bias_gt, measured_noise_mean, noise_variance, measured_noise_var, outlier_probability, linear_error, linear_error_full, linear_linear_bias_error, linear_constant_bias_error, error, error_full, constant_bias_error, linear_bias_error, error_cluster, error_cluster_full, constant_bias_error_cluster, linear_bias_error_cluster]

        # Open a CSV file in append mode
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

if __name__ == '__main__':
    

    # run_simulation_metrics()
    run_simulation_linear_least_squares_comparison()
    # run_simulation_non_linear_least_squares_comparison()
    # run_simulation_outlier_filtering_comparison()
    # run_simulation_clustering_comparison()
    # run_simulation_trajectory_optimisation_comparison()
    # run_simulation_residual_analysis()
    # run_simulation_error_debugging()
